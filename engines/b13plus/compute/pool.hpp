// StemDeck Android: a process-wide thread pool, so the heavy maths inside
// one segment can use every core.
//
// The engine separates several segments at once (one worker thread each,
// as many as memory allows). Each worker's large matrix products are cut
// into small column chunks that any idle core picks up, the calling worker
// included. Chunks are handed out one at a time, so fast and slow cores
// (big.LITTLE) each take as many as they can.
#pragma once

#include <cstddef>
#include <functional>

namespace sdpool
{
// Total threads doing maths, the callers included. 0 = all cores.
void set_threads(int n);
int threads();

// Runs fn(i) for i in [0, n), spread over the pool. Returns when all done.
// Safe to call from several threads at once, and from inside a task (then
// it runs inline).
void parallel_for(int n, const std::function<void(int)> &fn);
} // namespace sdpool
