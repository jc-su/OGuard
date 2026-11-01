// Test the month_start CUDA kernel on the GPU against a CPU reference.
// The reference walks the proleptic Gregorian calendar month by month from
// 0001-01-01 to 9999-12-31 and records the first day of each month. Besides the
// required sizes, the test checks every day of the stated range.
// Build: nvcc -O3 -arch=sm_86 -o test test.cu
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "kernel.cu"

#define CHECK_CUDA(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                   \
                         cudaGetErrorString(err_));                                   \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

#define REQUIRE(cond)                                                                 \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,    \
                         #cond);                                                      \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

constexpr int kThreads = 256;
constexpr int32_t kFirstDay = -719162;  // 0001-01-01
constexpr int32_t kLastDay = 2932896;   // 9999-12-31

template <typename T>
T* to_device(const std::vector<T>& host) {
    T* device = nullptr;
    CHECK_CUDA(cudaMalloc(&device, std::max<size_t>(host.size(), 1) * sizeof(T)));
    if (!host.empty()) {
        CHECK_CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
    return device;
}

template <typename T>
std::vector<T> to_host(const T* device, size_t size) {
    std::vector<T> host(size);
    if (size != 0) {
        CHECK_CUDA(cudaMemcpy(host.data(), device, size * sizeof(T), cudaMemcpyDeviceToHost));
    }
    return host;
}

// month_start_table[d - kFirstDay] is the first day of the month that contains day d.
std::vector<int32_t> month_start_table() {
    std::vector<int32_t> table(kLastDay - kFirstDay + 1);
    int32_t day = kFirstDay;
    for (int year = 1; year <= 9999; ++year) {
        const bool leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        const int lengths[12] = {31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        for (int month = 0; month < 12; ++month) {
            for (int d = 0; d < lengths[month]; ++d) table[day + d - kFirstDay] = day;
            day += lengths[month];
        }
    }
    REQUIRE(day == kLastDay + 1);
    return table;
}

void check(const std::vector<int32_t>& days, const std::vector<int32_t>& table) {
    const int64_t n = static_cast<int64_t>(days.size());
    std::vector<int32_t> expected(n);
    for (int64_t i = 0; i < n; ++i) expected[i] = table[days[i] - kFirstDay];

    int32_t* d_days = to_device(days);
    int32_t* d_out = to_device(std::vector<int32_t>(n, -1));
    if (n > 0) {
        const int64_t blocks = (n + kThreads - 1) / kThreads;
        kernel<<<static_cast<unsigned>(blocks), kThreads>>>(d_days, d_out, n);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    REQUIRE(to_host(d_out, n) == expected);

    CHECK_CUDA(cudaFree(d_days));
    CHECK_CUDA(cudaFree(d_out));
}

int main() {
    const std::vector<int32_t> table = month_start_table();
    REQUIRE(table[0 - kFirstDay] == 0);
    std::mt19937_64 rng(8);
    std::uniform_int_distribution<int32_t> any_day(kFirstDay, kLastDay);
    for (int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
        std::vector<int32_t> days(n);
        for (int32_t& d : days) d = any_day(rng);
        check(days, table);
    }
    std::vector<int32_t> all_days(kLastDay - kFirstDay + 1);
    for (size_t i = 0; i < all_days.size(); ++i) all_days[i] = kFirstDay + static_cast<int32_t>(i);
    check(all_days, table);
    std::printf("month_start cuda: all tests passed\n");
    return 0;
}
