// StemDeck Android: optional per-operation timing for the benchmark app.
//
// SD_PROF("name") times the rest of the enclosing scope and adds it to a
// global counter for that name. It compiles to nothing unless SD_PROFILE is
// defined, so the shipping app pays nothing. Scopes are placed on leaf
// operations only, so the totals do not double count.
#pragma once

#ifdef SD_PROFILE
#include <string>

namespace sdprof
{
int reg(const char *name);
void add(int id, long long ns);
long long now_ns();
void reset();
// One line per operation: name, calls, total ms; sorted by total.
std::string report();

struct Scope
{
    int id;
    long long t0;
    explicit Scope(int i) : id(i), t0(now_ns()) {}
    ~Scope() { add(id, now_ns() - t0); }
};
} // namespace sdprof

#define SD_PROF_CAT2(a, b) a##b
#define SD_PROF_CAT(a, b) SD_PROF_CAT2(a, b)
#define SD_PROF(name)                                                          \
    static const int SD_PROF_CAT(sdprof_id_, __LINE__) = sdprof::reg(name);    \
    sdprof::Scope SD_PROF_CAT(sdprof_scope_, __LINE__)(                         \
        SD_PROF_CAT(sdprof_id_, __LINE__))
// For spans that do not match a C++ scope.
#define SD_PROF_BEGIN(v, name)                                                 \
    static const int v##_sdprof_id = sdprof::reg(name);                         \
    long long v##_sdprof_t0 = sdprof::now_ns()
#define SD_PROF_END(v) sdprof::add(v##_sdprof_id, sdprof::now_ns() - v##_sdprof_t0)
#else
#define SD_PROF(name) ((void)0)
#define SD_PROF_BEGIN(v, name) ((void)0)
#define SD_PROF_END(v) ((void)0)
#endif
