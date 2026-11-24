// Copyright 2026 Pierre Talbot

#include <cstdio>
#include <climits>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
#include <new>
#include "battery/allocator.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/vector.hpp"
#include "lala/fixpoint.hpp"

namespace bt = ::battery;
using namespace lala;

static int failures = 0;
#define CHECK(c) do { if(!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++failures; } } while(0)

constexpr int TPB = 256;

static_assert(std::is_empty_v<BlockAsynchronousFixpointGPU>);
static_assert(std::is_empty_v<WarpAsynchronousFixpointGPU>);
static_assert(std::is_empty_v<FixpointEngine<bt::WarpsOfBlock>>);
static_assert(DynamicSchedule<ActiveSetGPU<bt::global_allocator, TPB>>);

template <class T>
T* managed_array(size_t n) {
  T* ptr;
  CUDAEX(cudaMallocManaged(&ptr, n * sizeof(T)));
  memset(ptr, 0, n * sizeof(T));
  return ptr;
}

/** I. Minimum of an array. */

struct MinResult {
  int min;
  int it1;      // Iterations of the first fixpoint (thread 0).
  int it2;      // Iterations of a second fixpoint, called right after the first one without barrier.
  int it3;      // Iterations of a third fixpoint, right after a fixpoint that ended after one iteration (grid only).
  int disagree; // Number of threads whose number of iterations differs from the one of thread 0.
};

__global__ void block_min_kernel(const int* v, int n, MinResult* r) {
  __shared__ int m;
  if(threadIdx.x == 0) {
    m = INT_MAX;
  }
  __syncthreads();
  BlockAsynchronousFixpointGPU fp;
  auto f = [&](int i) { int old = atomicMin(&m, v[i]); return v[i] < old; };
  int it1 = fp.fixpoint(n, f);
  int it2 = fp.fixpoint(n, f);
  if(threadIdx.x == 0) {
    r->min = m;
    r->it1 = it1;
    r->it2 = it2;
  }
  __syncthreads();
  if(it1 != r->it1) {
    atomicAdd(&r->disagree, 1);
  }
}

__global__ void grid_min_kernel(const int* v, int n, MinResult* r, int* m) {
  auto grid = cooperative_groups::this_grid();
  bt::unique_ptr<GridAsynchronousFixpointGPU, bt::global_allocator> fp_ptr;
  GridAsynchronousFixpointGPU& fp = bt::make_unique_grid(fp_ptr);
  auto f = [&](int i) { int old = atomicMin(m, v[i]); return v[i] < old; };
  int it1 = fp.fixpoint(n, f);
  int it2 = fp.fixpoint(n, f);
  int it3 = fp.fixpoint(n, f);
  if(grid.thread_rank() == 0) {
    r->min = *m;
    r->it1 = it1;
    r->it2 = it2;
    r->it3 = it3;
  }
  grid.sync();
  if(it1 != r->it1) {
    atomicAdd(&r->disagree, 1);
  }
  grid.sync(); // All the threads are done with `fp` before `fp_ptr` is destroyed.
}

/** Same as `grid_min_kernel` with a single fixpoint, but on an engine allocated by the host and reused across launches. */
__global__ void grid_min_reuse_kernel(const int* v, int n, GridAsynchronousFixpointGPU* fp, MinResult* r, int* m) {
  auto grid = cooperative_groups::this_grid();
  auto f = [&](int i) { int old = atomicMin(m, v[i]); return v[i] < old; };
  int it1 = fp->fixpoint(n, f);
  if(grid.thread_rank() == 0) {
    r->min = *m;
    r->it1 = it1;
  }
  grid.sync();
  if(it1 != r->it1) {
    atomicAdd(&r->disagree, 1);
  }
}

void check_min(const MinResult& r, int expected, int blocks, int threads) {
  printf("minimum on %d block(s) of %d thread(s): iterations %d then %d\n", blocks, threads, r.it1, r.it2);
  CHECK(r.min == expected);
  CHECK(r.it1 == 2); // The first iteration finds the minimum, the second detects the fixpoint.
  CHECK(r.it2 == 1); // Already a fixpoint.
  CHECK(r.disagree == 0);
}

void test_minimum() {
  const int n = 1000000;
  int* v = managed_array<int>(n);
  std::mt19937 gen{42};
  std::uniform_int_distribution<int> dist{-10000000, 10000000};
  for(int i = 0; i < n; ++i) {
    v[i] = dist(gen);
  }
  int expected = *std::min_element(v, v + n);
  MinResult* r = managed_array<MinResult>(1);

  for(int threads : {1, 2, 100, 128, 256}) {
    memset(r, 0, sizeof(MinResult));
    block_min_kernel<<<1, threads>>>(v, n, r);
    CUDAEX(cudaDeviceSynchronize());
    check_min(*r, expected, 1, threads);
  }

  int dev = 0, supports_coop = 0, num_sms = 0;
  CUDAEX(cudaDeviceGetAttribute(&supports_coop, cudaDevAttrCooperativeLaunch, dev));
  CUDAEX(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
  if(!supports_coop) {
    printf("Device does not support cooperative launch, grid tests skipped.\n");
  }
  else {
    int* m = managed_array<int>(1);
    // Reused by all the launches below, whatever their shape: its flags rotate across the launches.
    GridAsynchronousFixpointGPU* shared_fp = managed_array<GridAsynchronousFixpointGPU>(1);
    new(shared_fp) GridAsynchronousFixpointGPU();
    for(int blocks : {2, 16, 48}) {
      for(int threads : {1, 100, 256}) {
        int blocks_per_sm = 0, reuse_blocks_per_sm = 0;
        CUDAEX(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, grid_min_kernel, threads, 0));
        CUDAEX(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&reuse_blocks_per_sm, grid_min_reuse_kernel, threads, 0));
        if(blocks > std::min(blocks_per_sm, reuse_blocks_per_sm) * num_sms) {
          printf("minimum on %d blocks of %d threads: skipped, the blocks cannot be co-resident.\n", blocks, threads);
          continue;
        }
        memset(r, 0, sizeof(MinResult));
        *m = INT_MAX;
        void* args[] = {&v, (void*)&n, &r, &m};
        CUDAEX(cudaLaunchCooperativeKernel((void*)grid_min_kernel, dim3(blocks), dim3(threads), args));
        CUDAEX(cudaDeviceSynchronize());
        check_min(*r, expected, blocks, threads);
        CHECK(r->it3 == 1);
        // From `INT_MAX`, then from the minimum, in two launches.
        *m = INT_MAX;
        for(int expected_it : {2, 1}) {
          memset(r, 0, sizeof(MinResult));
          void* reuse_args[] = {&v, (void*)&n, &shared_fp, &r, &m};
          CUDAEX(cudaLaunchCooperativeKernel((void*)grid_min_reuse_kernel, dim3(blocks), dim3(threads), reuse_args));
          CUDAEX(cudaDeviceSynchronize());
          CHECK(r->min == expected);
          CHECK(r->it1 == expected_it);
          CHECK(r->disagree == 0);
        }
      }
    }
    CUDAEX(cudaFree(shared_fp));
    CUDAEX(cudaFree(m));
  }
  CUDAEX(cudaFree(r));
  CUDAEX(cudaFree(v));
}

/** II. Stop condition: every function always changes something, only `must_stop` ends the fixpoint. */

__global__ void block_stop_kernel(int n, int* x_out, int* it_out) {
  __shared__ int x;
  if(threadIdx.x == 0) {
    x = 0;
  }
  __syncthreads();
  int it = BlockAsynchronousFixpointGPU{}.fixpoint(n,
    [&](int) { atomicAdd(&x, 1); return true; },
    [&]{ return atomicAdd(&x, 0) >= 3 * n; });
  if(threadIdx.x == 0) {
    *x_out = x;
    *it_out = it;
  }
}

void test_stop() {
  const int n = 100003;
  int* out = managed_array<int>(2);
  block_stop_kernel<<<1, TPB>>>(n, &out[0], &out[1]);
  CUDAEX(cudaDeviceSynchronize());
  printf("stop: x=%d after %d iterations (3n = %d)\n", out[0], out[1], 3 * n);
  // The stop condition holds after the third iteration, and it may be observed one iteration late.
  CHECK(out[1] == 3 || out[1] == 4);
  CHECK(out[0] == out[1] * n);
  CUDAEX(cudaFree(out));
}

/** III. ActiveSetGPU: the function `i` counts up to `targets[i]`, then retires itself. */

struct SetResult {
  int ok, iterations, size, calls_after_retire;
  int reset_size, retire_size, retire2_size, content_ok;
  int view_iterations, view_size, view_refined_size, view_content_ok;
};

/** Retire the functions `i` for which `g(i)` is `false`, outside of a fixpoint, with all the threads of the block. */
template <class ActiveSet, class G>
__device__ void retire_unless(ActiveSet& active, const G& g) {
  for(int k = threadIdx.x; k < active.size(); k += blockDim.x) {
    if(!g(active[k])) {
      active.deactivate(active[k]);
    }
  }
  // `refine` must be called after a barrier following all the deactivations.
  __syncthreads();
  active.refine();
  __syncthreads(); // `refine` does not synchronize on exit.
}

__global__ void active_set_kernel(int n, int* counters, const int* targets, int* calls_after_retire, SetResult* r) {
  using ActiveSet = ActiveSetGPU<bt::global_allocator, TPB>;
  bt::unique_ptr<bt::vector<int, bt::global_allocator>, bt::global_allocator> scratchpad_ptr;
  auto& scratchpad = bt::make_unique_block(scratchpad_ptr, ActiveSet::scratchpad_size());
  bt::unique_ptr<ActiveSet, bt::global_allocator> active_ptr;
  ActiveSet& active = bt::make_unique_block(active_ptr, n, scratchpad);
  int iterations = BlockAsynchronousFixpointGPU{}.fixpoint(active, [&](int i) {
    if(!active.is_active(i)) {
      atomicAdd(&calls_after_retire[i], 1);
    }
    bool changed = false;
    if(counters[i] < targets[i]) {
      counters[i]++;
      changed = true;
    }
    if(counters[i] == targets[i]) {
      active.deactivate(i);
    }
    return changed;
  });
  if(threadIdx.x == 0) {
    r->iterations = iterations;
    r->size = active.size();
    r->ok = 1;
    for(int i = 0; i < n; ++i) {
      r->ok &= counters[i] == targets[i];
      r->calls_after_retire += calls_after_retire[i];
    }
  }
  __syncthreads(); // The thread 0 is done reading the active set before `reset`.
  active.reset();
  __syncthreads(); // `reset` does not synchronize on exit.
  if(threadIdx.x == 0) {
    r->reset_size = active.size();
  }
  retire_unless(active, [](int i) { return i % 2 == 0; });
  if(threadIdx.x == 0) {
    r->retire_size = active.size();
  }
  retire_unless(active, [](int i) { return i % 4 == 0; });
  if(threadIdx.x == 0) {
    r->retire2_size = active.size();
    r->content_ok = 1;
    for(int k = 0; k < active.size(); ++k) {
      r->content_ok &= active[k] % 4 == 0 && active.is_active(active[k]);
    }
  }
  // Deactivate during the fixpoint, but refine ourselves after the fixpoint.
  __syncthreads(); // The thread 0 is done checking the content of the active set before `reset`.
  active.reset();
  __syncthreads();
  int view_iterations = BlockAsynchronousFixpointGPU{}.fixpoint(static_view(active), [&](int i) {
    if(i % 3 != 0) {
      active.deactivate(i);
    }
    return false;
  });
  if(threadIdx.x == 0) {
    r->view_iterations = view_iterations;
    r->view_size = active.size();
  }
  active.refine();
  __syncthreads();
  if(threadIdx.x == 0) {
    r->view_refined_size = active.size();
    r->view_content_ok = 1;
    for(int k = 0; k < active.size(); ++k) {
      r->view_content_ok &= active[k] % 3 == 0 && active.is_active(active[k]);
    }
  }
  __syncthreads(); // All the threads are done with `active` before it is destroyed.
}

void test_active_set() {
  const int n = 1000; // Not a multiple of TPB.
  int* counters = managed_array<int>(n);
  int* targets = managed_array<int>(n);
  int* calls_after_retire = managed_array<int>(n);
  SetResult* r = managed_array<SetResult>(1);
  for(int i = 0; i < n; ++i) {
    targets[i] = i % 5;
  }
  active_set_kernel<<<1, TPB>>>(n, counters, targets, calls_after_retire, r);
  CUDAEX(cudaDeviceSynchronize());
  printf("active set: %d iterations; reset %d, retire %d, retire %d, content ok=%d\n",
    r->iterations, r->reset_size, r->retire_size, r->retire2_size, r->content_ok);
  CHECK(r->ok == 1);
  CHECK(r->iterations == 5); // Four iterations change something, the fifth one has no function left.
  CHECK(r->size == 0);
  CHECK(r->calls_after_retire == 0);
  CHECK(r->reset_size == n);
  CHECK(r->retire_size == n / 2);
  CHECK(r->retire2_size == n / 4);
  CHECK(r->content_ok == 1);
  printf("active set with static_view: %d iteration(s), size %d, then %d after refine, content ok=%d\n",
    r->view_iterations, r->view_size, r->view_refined_size, r->view_content_ok);
  CHECK(r->view_iterations == 1);
  CHECK(r->view_size == n);                // Not refined during the fixpoint.
  CHECK(r->view_refined_size == (n + 2) / 3); // The multiples of 3 in [0, n).
  CHECK(r->view_content_ok == 1);
  CUDAEX(cudaFree(r));
  CUDAEX(cudaFree(calls_after_retire));
  CUDAEX(cudaFree(targets));
  CUDAEX(cudaFree(counters));
}

/** IV. Nested fixpoint: a block of warps, each warp computes a local fixpoint on a chunk of 32 functions.
 * The function `i` counts up to `targets[i]`. */

__global__ void nested_kernel(int n, int* counters, const int* targets, int* outer_iterations) {
  FixpointEngine<bt::WarpsOfBlock> fp_warps;
  WarpAsynchronousFixpointGPU fp_warp;
  int lane = threadIdx.x % 32;
  int it = fp_warps.fixpoint((n + 31) / 32, [&](int c) {
    int first = 32 * c;
    int m = min(32, n - first);
    int i = first + lane;
    int target = lane < m ? targets[i] : 0; // Loaded once per lane.
    // Lanes `>= m` do not call `f`, but participate in the reduction of the warp.
    return fp_warp.fixpoint(m, [&](int) {
      if(counters[i] < target) {
        counters[i]++;
        return true;
      }
      return false;
    }) > 1;
  });
  if(threadIdx.x == 0) {
    *outer_iterations = it;
  }
}

void test_nested() {
  const int n = 1000; // Not a multiple of 32.
  int* counters = managed_array<int>(n);
  int* targets = managed_array<int>(n);
  int* outer_iterations = managed_array<int>(1);
  for(int i = 0; i < n; ++i) {
    targets[i] = i % 7;
  }
  nested_kernel<<<1, TPB>>>(n, counters, targets, outer_iterations);
  CUDAEX(cudaDeviceSynchronize());
  printf("nested: %d outer iterations\n", *outer_iterations);
  bool ok = true;
  for(int i = 0; i < n; ++i) {
    ok &= counters[i] == targets[i];
  }
  CHECK(ok);
  CHECK(*outer_iterations == 2); // The warps reach their local fixpoint in the first outer iteration.
  CUDAEX(cudaFree(outer_iterations));
  CUDAEX(cudaFree(targets));
  CUDAEX(cudaFree(counters));
}

int main() {
  test_minimum();
  test_stop();
  test_active_set();
  test_nested();
  printf("%s (%d failure(s))\n", failures == 0 ? "OK" : "FAILED", failures);
  return failures == 0 ? 0 : 1;
}
