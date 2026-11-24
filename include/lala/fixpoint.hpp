// Copyright 2022 Pierre Talbot

#ifndef LALA_FIXPOINT_FIXPOINT_HPP
#define LALA_FIXPOINT_FIXPOINT_HPP

#include <type_traits>
#include "battery/utility.hpp"
#include "battery/group.hpp"
#include "lala/schedule.hpp"

namespace lala {

/** The default stop condition: never stop before reaching a fixpoint. */
struct never_stop {
  CUDA INLINE constexpr bool operator()() const { return false; }
};

/** A fixpoint engine executing the schedule with the threads of `Group`. */
template <class Group>
class FixpointEngine {
public:
  using group_type = Group;

private:
  /** Empty for all groups but `GridGroup`, hence the engine is empty too. */
  [[no_unique_address]] Group group;

public:
  FixpointEngine() = default;
  CUDA explicit FixpointEngine(const Group& group): group(group) {}

  /** One iteration: the current thread executes `f(s[k])` for `k = rank, rank + size, ...` where `rank` and `size` are those of the group.
   * \return `true` if the current thread changed anything, `false` otherwise. Reduction must be performed by the caller. */
  template <class S, class F>
    requires Schedule<S>
  CUDA INLINE bool iterate(const S& s, const F& f) const {
    bool changed = false;
    for(int k = group.rank(); k < s.size(); k += group.size()) {
      changed |= f(s[k]);
    }
    return changed;
  }

  /** Iterate `f` on the schedule `s` until we reach a fixpoint or `must_stop()`.
   * It must be called by all the threads of the group.
   * \return The number of iterations. */
  template <class S, class F, class Stop = never_stop>
    requires Schedule<std::remove_cvref_t<S>>
  CUDA int fixpoint(S&& s, const F& f, const Stop& must_stop = Stop{}) {
    int iterations = 0;
    bool cont;
    do {
      cont = iterate(s, f) && !must_stop();
      cont = group.any_sync(cont);
      ++iterations;
      if constexpr(DynamicSchedule<std::remove_cvref_t<S>>) {
        if(cont) {
          s.refine();
          // `refine` does not synchronize on exit, but the next iteration reads the refined schedule.
          group.sync();
        }
      }
    } while(cont);
    return iterations;
  }

  /** Same as `fixpoint(AllOf{n}, f, must_stop)`. */
  template <class F, class Stop = never_stop>
  CUDA int fixpoint(int n, const F& f, const Stop& must_stop = Stop{}) {
    return fixpoint(AllOf{n}, f, must_stop);
  }
};

/** Sequential fixpoint: the functions are composed by functional composition \f$ f = f_{n-1} \circ \ldots \circ f_0 \f$.
 * This strategy corresponds to the Gauss-Seidel iteration method. */
using GaussSeidelIteration = FixpointEngine<battery::Sequential>;

#ifdef __CUDACC__

/** Parallel fixpoint over a block: the functions are composed by parallel composition \f$ f = f_0 \| \ldots \| f_{n-1} \f$. */
using BlockAsynchronousFixpointGPU = FixpointEngine<battery::ThisBlock>;

/** Same as `BlockAsynchronousFixpointGPU` over the lanes of a warp. */
using WarpAsynchronousFixpointGPU = FixpointEngine<battery::ThisWarp>;

/** Same as `BlockAsynchronousFixpointGPU` over the whole grid.
 * Unlike the other engines, it holds data (see `battery::GridGroup`): allocate it in global memory (e.g., with `battery::make_unique_grid`), and share it among the blocks. */
using GridAsynchronousFixpointGPU = FixpointEngine<battery::GridGroup>;

#endif // __CUDACC__

} // namespace lala

#endif
