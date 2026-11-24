// Copyright 2026 Pierre Talbot

#ifndef LALA_FIXPOINT_SCHEDULE_HPP
#define LALA_FIXPOINT_SCHEDULE_HPP

#include <cassert>
#include <concepts>
#include "battery/utility.hpp"
#include "battery/allocator.hpp"
#include "battery/vector.hpp"
#include "battery/group.hpp"

namespace lala {

/** A schedule tells a fixpoint engine which functions to execute during one iteration:
 * the functions `s[0], ..., s[s.size()-1]`.
 * The engine reads `size()` and `operator[]` again at every iteration. */
template <class S>
concept Schedule = requires(const S& s, int k) {
  { s.size() } -> std::convertible_to<int>;
  { s[k] } -> std::convertible_to<int>;
};

/** A dynamic schedule is a schedule that changes between two iterations.
 * The engine calls `refine()` between two iterations, with all the threads of its group, but not after the last iteration: the functions deactivated during the last iteration stay in the schedule until the next `refine()`.
 * The engine calls `refine()` after a barrier, and synchronizes its group after `refine()`, so `refine()` does not need to start or end with a barrier. */
template <class S>
concept DynamicSchedule = Schedule<S> && requires(S& s) { s.refine(); };

/** The static schedule of all the functions `0, ..., n-1`. */
struct AllOf {
  int n;
  CUDA INLINE constexpr int size() const { return n; }
  CUDA INLINE constexpr int operator[](int k) const { return k; }
};

/** A static view of the schedule `s`: the fixpoint engine never calls `s.refine()` on it.
 * Use it to keep a dynamic schedule unchanged during a fixpoint, and to refine it yourself afterwards.
 * The functions deactivated during the fixpoint keep being executed until you call `s.refine()`.
 * \code
 * fp.fixpoint(static_view(active), [&](int i) {
 *   bool changed = deduce(i);
 *   if(entailed(i)) { active.deactivate(i); }
 *   return changed;
 * });
 * active.refine(); // The fixpoint ends with a barrier, so we can refine right away.
 * __syncthreads(); // On GPU: `refine` does not synchronize on exit.
 * \endcode
 */
template <class S>
class StaticView {
  const S& s;
public:
  CUDA explicit StaticView(const S& s): s(s) {}
  CUDA INLINE int size() const { return s.size(); }
  CUDA INLINE int operator[](int k) const { return s[k]; }
};

template <class S>
  requires Schedule<S>
CUDA StaticView<S> static_view(const S& s) {
  return StaticView<S>(s);
}

/** A dynamic schedule over the functions `0, ..., n-1`, from which functions can be retired.
 * During an iteration, any function can retire a function `i` (usually itself) by calling `deactivate(i)`.
 * The function `i` is removed from the schedule by `refine()`, at the end of the iteration.
 *
 * This version is sequential: use it with `GaussSeidelIteration`.
 *
 * Example, removing the propagators that are entailed:
 * \code
 * ActiveSetCPU<> active(n);
 * GaussSeidelIteration{}.fixpoint(active, [&](int i) {
 *   bool changed = deduce(i);
 *   if(entailed(i)) { active.deactivate(i); }
 *   return changed;
 * });
 * \endcode
 */
template <class Allocator = battery::standard_allocator>
class ActiveSetCPU {
public:
  using allocator_type = Allocator;

private:
  /** The functions `indexes[0..n)` are active, the functions `indexes[n..)` are retired. */
  battery::vector<int, allocator_type> indexes;

  /** `active[i]` is `false` if the function `i` was deactivated. */
  battery::vector<bool, allocator_type> active;

  /** The number of active functions. */
  int n;

  /** `true` if a function was deactivated since the last `refine()`. */
  bool dirty;

public:
  /** Activate the functions `0, ..., n-1`. */
  CUDA ActiveSetCPU(int n, const allocator_type& alloc = allocator_type())
   : indexes(n, alloc), active(n, true, alloc), n(n), dirty(false)
  {
    for(int i = 0; i < n; ++i) {
      indexes[i] = i;
    }
  }

  /** \return The number of active functions. */
  CUDA int size() const { return n; }

  /** \return The `k`-th active function. */
  CUDA int operator[](int k) const { return indexes[k]; }

  CUDA bool is_active(int i) const { return active[i]; }

  /** Retire the function `i`, it is removed from the schedule at the next `refine()`. */
  CUDA void deactivate(int i) {
    active[i] = false;
    dirty = true;
  }

  /** Remove the deactivated functions from the schedule. */
  CUDA void refine() {
    if(!dirty) {
      return;
    }
    for(int k = 0; k < n; ) {
      if(!active[indexes[k]]) {
        battery::swap(indexes[k], indexes[--n]);
      }
      else {
        ++k;
      }
    }
    dirty = false;
  }

  /** Activate all the functions again. */
  CUDA void reset() {
    n = static_cast<int>(indexes.size());
    for(int i = 0; i < n; ++i) {
      active[i] = true;
    }
    dirty = false;
  }
};

#ifdef __CUDACC__

/** Same as `ActiveSetCPU`, but for block-wide fixpoint engines (`BlockAsynchronousFixpointGPU` and `FixpointEngine<battery::WarpsOfBlock>`).
 * `refine()` compacts the active functions with all the threads of the block.
 * \tparam TPB The number of threads per block, it must be equal to `blockDim.x` and a multiple of 32.
 * \tparam ScratchpadAllocator The allocator of the scratchpad.
 *
 * Scratchpad: a vector of integers owned by the caller and given by reference to the constructor, so it can be shared among several algorithms.
 * Its size must be at least `scratchpad_size()`.
 * `refine()` uses it, so it must not be used in parallel elsewhere while `refine()` runs, and other uses must be separated from `refine()` by a barrier.
 * For efficiency, its data should be in shared memory.
 *
 * The object must be shared by all the threads of the block: construct it with `battery::make_unique_block`.
 *
 * `refine()` and `reset()` synchronize the threads neither on entry nor on exit.
 * Before calling them, make sure all the threads have exited the other methods of this object (e.g., with a barrier).
 * After calling them yourself, call `__syncthreads()` before reading the active set.
 *
 * Example:
 * \code
 * using ActiveSet = ActiveSetGPU<bt::global_allocator, TPB>;
 * bt::unique_ptr<bt::vector<int, bt::global_allocator>, bt::global_allocator> scratchpad_ptr;
 * auto& scratchpad = bt::make_unique_block(scratchpad_ptr, ActiveSet::scratchpad_size());
 * bt::unique_ptr<ActiveSet, bt::global_allocator> active_ptr;
 * ActiveSet& active = bt::make_unique_block(active_ptr, n, scratchpad);
 * BlockAsynchronousFixpointGPU{}.fixpoint(active, [&](int i) {
 *   bool changed = deduce(i);
 *   if(entailed(i)) { active.deactivate(i); }
 *   return changed;
 * });
 * active.reset();  // No barrier needed before: the fixpoint ends with a barrier, after which the active set is not read.
 * __syncthreads(); // `reset` does not synchronize on exit.
 * // ... use `active` again ...
 * __syncthreads(); // All the threads must be done with `active` before `active_ptr` is destroyed.
 * \endcode
 */
template <class Allocator, int TPB, class ScratchpadAllocator = Allocator>
class ActiveSetGPU {
  static_assert(TPB % 32 == 0, "The number of threads per block must be a multiple of 32.");

public:
  using allocator_type = Allocator;
  using scratchpad_type = battery::vector<int, ScratchpadAllocator>;

  /** The minimal size of the scratchpad: the number of active functions per warp in a chunk, double-buffered (see `refine()`). */
  CUDA INLINE static constexpr int scratchpad_size() { return 2 * (TPB / 32); }

private:
  using ivector = battery::vector<int, allocator_type>;
  using bvector = battery::vector<bool, allocator_type>;

  /** The indexes of the active functions are `indexes[0..count)`.
   * The size of `indexes` is always `n`: we never resize it. */
  ivector indexes;

  /** The number of active functions. */
  int count;

  /** `active[i]` is `false` if the function `i` was deactivated. */
  bvector active;

  /** `true` if a function was deactivated since the last `refine()`. */
  bool dirty;

  /** The scratchpad owned by the caller, of size at least `scratchpad_size()`. */
  scratchpad_type& scratchpad;

public:
  /** Activate the functions `0, ..., n-1`.
   * It is executed by a single thread (e.g., by `battery::make_unique_block`).
   * The allocator is passed last, as expected by `battery::allocate_unique`. */
  __device__ ActiveSetGPU(int n, scratchpad_type& scratchpad, const allocator_type& alloc = allocator_type())
   : indexes(n, alloc), count(n), active(n, true, alloc), dirty(false), scratchpad(scratchpad)
  {
    assert(scratchpad.size() >= scratchpad_size());
    for(int i = 0; i < n; ++i) {
      indexes[i] = i;
    }
  }

  /** \return The number of active functions. */
  CUDA int size() const { return count; }

  /** \return The `k`-th active function. */
  CUDA int operator[](int k) const { return indexes[k]; }

  CUDA bool is_active(int i) const { return active[i]; }

  /** Retire the function `i`, it is removed from the schedule at the next `refine()`.
   * Several threads can call it concurrently. */
  CUDA void deactivate(int i) {
    active[i] = false;
    dirty = true;
  }

  /** Remove the deactivated functions from the schedule, with all the threads of the block.
   *
   * Precondition: call it after a barrier following the last `deactivate()`, and do not call any other method until a barrier after `refine()`.
   * The fixpoint engine satisfies this precondition (it calls `refine()` between two barriers).
   * Hence, all the threads read the same value of `dirty`: it is only cleared by the thread 0 after the first barrier of the loop below, which all the threads reach after reading it.
   *
   * It does not synchronize on exit: call `__syncthreads()` before reading the refined schedule (the fixpoint engine does it for you).
   *
   * The compaction is done in place, by chunks of `TPB` functions:
   * indexes:   0 1 2 3   (the active functions)
   * active:    1 0 0 1   (by function index)
   * pos:       0 1 1 1   (exclusive prefix sum of `active[indexes[k]]`)
   * indexes:   0 3       (after compaction)
   * The position `pos` of an active function is the number of active functions in the previous warps of the chunk, plus the number of active functions in the previous lanes of its warp.
   * The lane 0 of each warp writes the number of active functions of its warp in the scratchpad (computed with a ballot), and after the barrier, the lane `l < warp` reads the one of the warp `l`, which is summed over the warp.
   * The barrier also returns the number of active functions of the chunk (`count_sync`).
   * The new position `carry + pos` of a function is never greater than its old position `k`, because `carry <= base` and `pos <= threadIdx.x`.
   * Hence, we never overwrite an index not yet read: those of the current chunk are read before the barrier preceding the writes, and those of the next chunks are after `base + TPB`.
   * In particular, `id` is kept in a register across the barrier: reading `indexes[k]` again after the barrier would race with the compaction.
   *
   * There is a single barrier per chunk because the scratchpad is double-buffered: the chunk `j` writes its warp totals in the half `j % 2` before its barrier, and reads them after.
   * The chunk `j + 2` overwrites this half only after the barrier of the chunk `j + 1`, which no thread reaches before reading the totals of the chunk `j`.
   *
   * We read `count` directly in the loop instead of a copy in a register: the compiler cannot unroll the loop (because `count` might change), which saves registers.
   * The thread 0 writes the new `count` after the barrier of the last chunk, so the only reads that might see it are the last tests `base < count`, for which `base >= count >= carry`: both values end the loop.
   */
  __device__ void refine() {
    assert(TPB == blockDim.x);
    if(!dirty) {
      return;
    }
    battery::ThisBlock block;
    battery::ThisWarp lanes;
    int warp = battery::WarpsOfBlock{}.rank();
    int lane = lanes.rank();
    int carry = 0; // The number of active functions in `indexes[0..base)`.
    for(int base = 0; base < count; base += TPB) {
      int k = base + block.rank();
      int id = k < count ? indexes[k] : -1;
      bool keep = k < count && active[id];
      unsigned int mask = lanes.ballot_sync(keep);
      int half = ((base / TPB) % 2) * (TPB / 32); // The half of the scratchpad used by this chunk.
      if(lane == 0) {
        scratchpad[half + warp] = battery::popcount(mask);
      }
      // The warp totals are visible, all the indexes of this chunk are read before we overwrite them, and `total` is the number of active functions in this chunk.
      int total = block.count_sync(keep);
      // The lane `l < warp` holds the number of active functions of the warp `l`.
      int prefix = lanes.reduce_add_sync(lane < warp ? scratchpad[half + lane] : 0);
      if(keep) {
        indexes[carry + prefix + battery::popcount(mask & ((1u << lane) - 1u))] = id;
      }
      carry += total;
    }
    if(threadIdx.x == 0) {
      // See above why the other threads can read `count` concurrently.
      count = carry;
      dirty = false;
    }
    // No barrier on exit: the caller synchronizes before reading the refined schedule (the engine does).
  }

  /** Activate all the functions again, with all the threads of the block.
   * Precondition: all the threads have exited the other methods of this object and finished reading the active set (e.g., after a barrier).
   * It does not synchronize on exit: call `__syncthreads()` before using the active set. */
  __device__ void reset() {
    int n = static_cast<int>(indexes.size());
    if(threadIdx.x == 0) {
      count = n;
      dirty = false;
    }
    for(int i = threadIdx.x; i < n; i += blockDim.x) {
      indexes[i] = i;
      active[i] = true;
    }
  }
};

#endif // __CUDACC__

} // namespace lala

#endif
