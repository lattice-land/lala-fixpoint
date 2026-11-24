// Copyright 2026 Pierre Talbot

#include <gtest/gtest.h>
#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>
#include "lala/fixpoint.hpp"

using namespace lala;

static_assert(Schedule<AllOf>);
static_assert(!DynamicSchedule<AllOf>);
static_assert(DynamicSchedule<ActiveSetCPU<>>);
static_assert(Schedule<StaticView<ActiveSetCPU<>>>);
static_assert(!DynamicSchedule<StaticView<ActiveSetCPU<>>>);
static_assert(std::is_empty_v<GaussSeidelIteration>);

/** `f(i)` updates `m` with the minimum of `m` and `v[i]`. */
struct Minimum {
  std::vector<int> v;
  int m = INT_MAX;
  bool operator()(int i) {
    if(v[i] < m) {
      m = v[i];
      return true;
    }
    return false;
  }
};

TEST(GaussSeidelIteration, Minimum) {
  Minimum min{{5, 3, 9, -2, 7, -2, 4}};
  GaussSeidelIteration fp;
  // The first iteration changes `m`, the second one detects the fixpoint.
  EXPECT_EQ(fp.fixpoint(static_cast<int>(min.v.size()), [&](int i) { return min(i); }), 2);
  EXPECT_EQ(min.m, -2);
  // We are already in a fixpoint.
  EXPECT_EQ(fp.fixpoint(static_cast<int>(min.v.size()), [&](int i) { return min(i); }), 1);
}

TEST(GaussSeidelIteration, ScheduleAndSizeOverload) {
  Minimum min{{5, 3, 9}};
  // `fixpoint(n, ...)` is the same as `fixpoint(AllOf{n}, ...)`, and accepts integers of other types.
  EXPECT_EQ(GaussSeidelIteration{}.fixpoint(AllOf{3}, [&](int i) { return min(i); }), 2);
  size_t n = min.v.size();
  EXPECT_EQ(GaussSeidelIteration{}.fixpoint(n, [&](int i) { return min(i); }), 1);
  EXPECT_EQ(min.m, 3);
}

TEST(GaussSeidelIteration, EmptySchedule) {
  int calls = 0;
  EXPECT_EQ(GaussSeidelIteration{}.fixpoint(0, [&](int) { ++calls; return true; }), 1);
  EXPECT_EQ(calls, 0);
}

TEST(GaussSeidelIteration, Iterate) {
  Minimum min{{5, 3, 9}};
  GaussSeidelIteration fp;
  EXPECT_TRUE(fp.iterate(AllOf{3}, [&](int i) { return min(i); }));
  EXPECT_FALSE(fp.iterate(AllOf{3}, [&](int i) { return min(i); }));
}

TEST(GaussSeidelIteration, Stop) {
  int x = 0;
  auto inc = [&](int) {
    if(x < 100) {
      ++x;
      return true;
    }
    return false;
  };
  // Each iteration adds 4 to `x`, we stop after the iteration reaching 12.
  EXPECT_EQ(GaussSeidelIteration{}.fixpoint(4, inc, [&]{ return x >= 10; }), 3);
  EXPECT_EQ(x, 12);
  // Without stop, we go until the fixpoint.
  EXPECT_EQ(GaussSeidelIteration{}.fixpoint(4, inc), 23);
  EXPECT_EQ(x, 100);
}

TEST(ActiveSetCPU, DeactivateDuringFixpoint) {
  // The function `i` counts up to `i % 4`, then retires itself.
  const int n = 10;
  std::vector<int> counters(n, 0);
  std::vector<int> calls_after_retire(n, 0);
  ActiveSetCPU<> active(n);
  int iterations = GaussSeidelIteration{}.fixpoint(active, [&](int i) {
    if(!active.is_active(i)) {
      calls_after_retire[i]++;
    }
    bool changed = false;
    if(counters[i] < i % 4) {
      counters[i]++;
      changed = true;
    }
    if(counters[i] == i % 4) {
      active.deactivate(i);
    }
    return changed;
  });
  for(int i = 0; i < n; ++i) {
    EXPECT_EQ(counters[i], i % 4);
    EXPECT_EQ(calls_after_retire[i], 0);
  }
  EXPECT_EQ(active.size(), 0);
  // Three iterations change something, the fourth one has no function left.
  EXPECT_EQ(iterations, 4);
}

TEST(ActiveSetCPU, RefineAfterFixpoint) {
  // Same as above, but we refine the active set ourselves after the fixpoint.
  const int n = 10;
  std::vector<int> counters(n, 0);
  std::vector<int> calls_after_retire(n, 0);
  ActiveSetCPU<> active(n);
  int iterations = GaussSeidelIteration{}.fixpoint(static_view(active), [&](int i) {
    if(!active.is_active(i)) {
      calls_after_retire[i]++;
    }
    bool changed = false;
    if(counters[i] < i % 4) {
      counters[i]++;
      changed = true;
    }
    if(counters[i] == i % 4) {
      active.deactivate(i);
    }
    return changed;
  });
  // Three iterations change something, the fourth one detects the fixpoint.
  EXPECT_EQ(iterations, 4);
  // The functions are deactivated but still in the schedule: they were executed in every iteration.
  EXPECT_EQ(active.size(), n);
  for(int i = 0; i < n; ++i) {
    EXPECT_EQ(counters[i], i % 4);
    EXPECT_FALSE(active.is_active(i));
    // The function `i` is deactivated during the iteration `max(1, i % 4)`.
    EXPECT_EQ(calls_after_retire[i], iterations - std::max(1, i % 4));
  }
  active.refine();
  EXPECT_EQ(active.size(), 0);
}

/** Retire the functions `i` for which `g(i)` is `false`, outside of a fixpoint. */
template <class G>
void retire_unless(ActiveSetCPU<>& active, const G& g) {
  for(int k = 0; k < active.size(); ++k) {
    if(!g(active[k])) {
      active.deactivate(active[k]);
    }
  }
  active.refine();
}

TEST(ActiveSetCPU, RefineReset) {
  const int n = 10;
  ActiveSetCPU<> active(n);
  EXPECT_EQ(active.size(), n);
  retire_unless(active, [](int i) { return i % 2 == 0; });
  EXPECT_EQ(active.size(), n / 2);
  for(int k = 0; k < active.size(); ++k) {
    EXPECT_EQ(active[k] % 2, 0);
  }
  retire_unless(active, [](int i) { return i % 4 == 0; });
  EXPECT_EQ(active.size(), 3);
  for(int k = 0; k < active.size(); ++k) {
    EXPECT_EQ(active[k] % 4, 0);
    EXPECT_TRUE(active.is_active(active[k]));
  }
  active.reset();
  EXPECT_EQ(active.size(), n);
  for(int i = 0; i < n; ++i) {
    EXPECT_TRUE(active.is_active(i));
  }
}

TEST(ActiveSetCPU, ResetWithPendingDeactivation) {
  const int n = 6;
  ActiveSetCPU<> active(n);
  // Deactivated but not removed yet: `reset` must activate it again.
  active.deactivate(2);
  active.reset();
  active.refine();
  EXPECT_EQ(active.size(), n);
  EXPECT_TRUE(active.is_active(2));
}
