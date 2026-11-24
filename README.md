# Fixpoint Library

This library computes the fixpoint of a set of monotone functions $f_0, \ldots, f_{n-1}$ over lattices, sequentially on CPU or in parallel on GPU.
It explores the programming model of [Talbot et al., AAAI 2022](https://ptal.github.io/papers/aaai2022.pdf): a parallel algorithm is written as a set of monotone functions, and their lock-free asynchronous iteration is correct by construction (see Cousot, 1977, and [Bertsekas and Tsitsiklis, 1989](https://web.mit.edu/dimitrib/www/pdc.html)).
Check also the library [lala-interval](https://github.com/lattice-land/lala-interval/) for standard lattices usable with this library.

For instance, the minimum of a vector `v` on a GPU block:

```cpp
#include "lala/ub.hpp"
#include "lala/fixpoint.hpp"
UB<int> m;  // shared by the threads of the block
int iterations = BlockAsynchronousFixpointGPU{}.fixpoint(n, [&](int i) { return m.meet(v[i]); });
```

## Fixpoint Engines

All engines are instances of `FixpointEngine<Group>`, where the group is the set of threads sharing the work.

| Engine | Threads | Iteration | Barrier per iteration |
| ------------- | ------------- | ------------- | ------------- |
| `GaussSeidelIteration` | the current thread (CPU or GPU) | $f_{n-1} \circ \ldots \circ f_0$ | none |
| `WarpAsynchronousFixpointGPU` | the 32 lanes of a warp | $f_0 \parallel \ldots \parallel f_{n-1}$ | `__any_sync` |
| `BlockAsynchronousFixpointGPU` | the threads of a block | $f_0 \parallel \ldots \parallel f_{n-1}$ | `__syncthreads_or` |
| `FixpointEngine<battery::WarpsOfBlock>` | the warps of a block (one function per warp) | $f_0 \parallel \ldots \parallel f_{n-1}$ | `__syncthreads_or` |
| `GridAsynchronousFixpointGPU` | the threads of a cooperative grid | $f_0 \parallel \ldots \parallel f_{n-1}$ | `__syncthreads_or` and `grid.sync()` |

The engines are empty objects, to declare as local variables, except the grid engine: allocate it in global memory and share it among the blocks (e.g., with `battery::make_unique_grid`).

### Interface

Let `fp` be an engine, `bool f(int i)` execute $f_i$ and return `true` if it changed something, and `s` a schedule (see below).

| Operation | Programming notation |
| ------------- | ------------- |
| One iteration over the schedule `s` | `fp.iterate(s, f)` |
| Fixpoint over the schedule `s` | `fp.fixpoint(s, f)` |
| Fixpoint over $f_0, \ldots, f_{n-1}$ | `fp.fixpoint(n, f)` |
| Fixpoint with an early stop | `fp.fixpoint(s, f, must_stop)` |

* The thread of rank `r` executes `f(s[k])` for `k = r, r + size, r + 2*size, ...`, where `size` is the number of threads of the group.
* `iterate` returns `true` if the current thread changed something; the result is not reduced over the group.
* `fixpoint` must be called by all the threads of the group, and returns the number of iterations (the same on all threads).
  Without stop, `iterations > 1` if and only if something changed.
* `must_stop()` is reduced in the same barrier as "changed", so it costs nothing extra, and nothing at all when omitted.
  Any thread may call it, so it must be visible to all the threads and remain `true` once it is `true` (e.g., `is_bot()`).
  The fixpoint then stops at most one iteration late, and the number of iterations does not tell whether a fixpoint was reached.

## Schedules

A schedule `s` is the list of functions executed in each iteration: `s.size()` and `s[k]`.
A dynamic schedule also has `s.refine()`, which the engine calls between two iterations.

| Schedule | Description |
| ------------- | ------------- |
| `AllOf{n}` | $f_0, \ldots, f_{n-1}$ |
| `ActiveSetCPU<Alloc>` | the active functions, sequential |
| `ActiveSetGPU<Alloc, TPB>` | the active functions, shared by a block of `TPB` threads |
| `static_view(s)` | `s`, but the engine never refines it |

An active set starts with all the functions active.
A function retires itself with `active.deactivate(i)` when it can no longer change anything (e.g., an entailed propagator), and `refine()` removes it from the schedule.

| Operation | Programming notation |
| ------------- | ------------- |
| Retire the function `i` | `active.deactivate(i)` |
| Is `i` still active? | `active.is_active(i)` |
| Remove the retired functions | `active.refine()` (called by the engine, or by you with `static_view`) |
| Activate all the functions | `active.reset()` |

`ActiveSetGPU` compacts in place with one barrier per chunk of `TPB` functions.
It needs a scratchpad of at least `scratchpad_size()` integers, preferably in shared memory, which can be shared with other algorithms when `refine()` is not running.

## Groups and Synchronization

A group provides `rank()`, `size()`, `any_sync(p)` (a barrier returning the logical or of `p`) and `sync()` (a plain barrier).
The groups `Sequential`, `ThisWarp`, `ThisBlock`, `WarpsOfBlock` and `GridGroup` are defined in `battery/group.hpp` of [cuda-battery](https://github.com/lattice-land/cuda-battery/).

Collective methods (e.g., `refine()` and `reset()`) synchronize neither on entry nor on exit:

* Add a barrier before reading their results.
* Before calling a method, make sure all the threads have exited the other methods of the same object.
* `fixpoint` ends with a barrier, so a method can be called right after it.
* The grid engine synchronizes at the start of each fixpoint, so consecutive grid fixpoints need no barrier in between.

## Nested Fixpoint

With `WarpsOfBlock`, the 32 lanes of a warp call `f(c)` together, so each warp can reach a local fixpoint on a chunk of 32 functions before the block-wide reduction.
`blockDim.x` must be a multiple of 32:

```cpp
FixpointEngine<battery::WarpsOfBlock> fp_warps;
WarpAsynchronousFixpointGPU fp_warp;
fp_warps.fixpoint((n + 31) / 32, [&](int c) {
  int first = 32 * c;
  int m = min(32, n - first);  // The lanes `>= m` do not call `f`, but still participate in the reduction.
  return fp_warp.fixpoint(m, [&](int i) { return f(first + i); }, must_stop) > 1;
}, must_stop);
```

## Plan

- [x] Import and clean fixpoint algorithms from lala-core.
- [x] Algorithm: finding the minimum in a sequence.
- [ ] Improve minimum algorithm to support computation over a grid.
- [ ] Algorithm: search an element in an unsorted list.
- [ ] Algorithm: union-find.
- [ ] Algorithm: breadth-first search algorithm.
- [ ] Algorithm: all-shortest paths algorithm (e.g. Floyd-Warshall).
- [ ] Combinatorial algorithm: [stable marriage problem](https://arxiv.org/pdf/2208.01370), redundant with Turbo?
- [ ] Combinatorial algorithm: [min-cut problems](https://arxiv.org/pdf/2512.18141), redundant with Turbo?
- [ ] Investigate: [stochastic gradient descent](https://proceedings.neurips.cc/paper_files/paper/2011/file/218a0aefd1d1a4be65601cc6ddc1520e-Paper.pdf).
- [ ] Investigate: [asynchronous algorithms](https://web.mit.edu/dimitrib/www/pdc.html)
- [ ] Benchmarking: Fixpoint algorithms VS optimized CUDA algorithms.

## Getting Started

To contribute, please fork the repository and send us your pull requests.
For the available lattices, you can check out [lala-interval](https://github.com/lattice-land/lala-interval/).

```
mkdir lattice-land
cd lattice-land
git clone <your fork of lala-fixpoint>
git clone git@github.com:lattice-land/cuda-battery.git
git clone git@github.com:lattice-land/lala-interval.git
cd <your fork of lala-fixpoint>
cmake --workflow --preset gpu-release-local --fresh
./build/gpu-release-local/minimum
```

You can add new binaries in `CMakeLists.txt`.

