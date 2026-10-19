// numpy's reductions and Python's rounding, where the last bit can decide a
// comparison downstream.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace sda {

// np.add.reduce: pairwise summation (numpy/_core/src/umath/loops_utils.h.src).
template <typename T>
T np_sum(const T* a, size_t n) {
    if (n < 8) {
        T res = 0;
        for (size_t i = 0; i < n; i++) res += a[i];
        return res;
    }
    if (n <= 128) {
        T r[8];
        for (int k = 0; k < 8; k++) r[k] = a[k];
        size_t i;
        for (i = 8; i < n - (n % 8); i += 8)
            for (int k = 0; k < 8; k++) r[k] += a[i + k];
        T res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_sum(a, n2) + np_sum(a + n2, n - n2);
}
template <typename T>
T np_sum(const std::vector<T>& a) { return np_sum(a.data(), a.size()); }
template <typename T>
T np_mean(const std::vector<T>& a) { return np_sum(a) / (T)a.size(); }
// np.std (ddof=0) for float64.
inline double np_std(const std::vector<double>& a) {
    double m = np_mean(a);
    std::vector<double> d(a.size());
    for (size_t i = 0; i < a.size(); i++) { double x = a[i] - m; d[i] = x * x; }
    return std::sqrt(np_mean(d));
}

// Python's round(x, ndigits): correctly rounded decimal, ties to even.
inline double py_round(double x, int ndigits) {
    if (!std::isfinite(x)) return x;
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*f", ndigits, x);
    double r = std::strtod(buf, nullptr);
    return r == 0 ? 0.0 * x : r;  // keeps -0.0 like Python
}
// Python's round(x) -> int (ties to even).
inline long py_round_int(double x) { return (long)std::nearbyint(x); }

// Shortest decimal that reads back as the same double (what Python's repr,
// and so json.dumps, writes), in JSON syntax.
inline std::string json_number(double x) {
    if (!std::isfinite(x)) return "null";
    if (x == std::floor(x) && std::fabs(x) < 1e15) {
        char b[32];
        std::snprintf(b, sizeof b, "%.1f", x);
        return b;
    }
    char buf[40];
    for (int p = 1; p <= 17; p++) {
        std::snprintf(buf, sizeof buf, "%.*g", p, x);
        if (std::strtod(buf, nullptr) == x) break;
    }
    return buf;
}

}  // namespace sda
