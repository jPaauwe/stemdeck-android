// See demucs/sdprof.hpp.
#ifdef SD_PROFILE
#include "sdprof.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <vector>

namespace sdprof
{
namespace
{
constexpr int kMax = 128;
std::mutex g_mutex;
const char *g_names[kMax];
int g_count = 0;
std::atomic<long long> g_ns[kMax];
std::atomic<long long> g_calls[kMax];
} // namespace

int reg(const char *name)
{
    std::lock_guard<std::mutex> lk(g_mutex);
    for (int i = 0; i < g_count; ++i)
        if (std::string(g_names[i]) == name)
            return i;
    if (g_count >= kMax)
        return kMax - 1;
    g_names[g_count] = name;
    g_ns[g_count] = 0;
    g_calls[g_count] = 0;
    return g_count++;
}

void add(int id, long long ns)
{
    g_ns[id].fetch_add(ns, std::memory_order_relaxed);
    g_calls[id].fetch_add(1, std::memory_order_relaxed);
}

long long now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void reset()
{
    std::lock_guard<std::mutex> lk(g_mutex);
    for (int i = 0; i < g_count; ++i)
    {
        g_ns[i] = 0;
        g_calls[i] = 0;
    }
}

std::string report()
{
    std::lock_guard<std::mutex> lk(g_mutex);
    std::vector<int> idx;
    for (int i = 0; i < g_count; ++i)
        if (g_calls[i].load() > 0)
            idx.push_back(i);
    std::sort(idx.begin(), idx.end(),
              [](int a, int b) { return g_ns[a].load() > g_ns[b].load(); });
    std::string out;
    char line[160];
    for (int i : idx)
    {
        std::snprintf(line, sizeof(line), "%s\t%lld\t%.1f\n", g_names[i],
                      g_calls[i].load(), g_ns[i].load() / 1e6);
        out += line;
    }
    return out;
}
} // namespace sdprof
#endif
