#include <cstdint>
#include <cstring>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int32_t* days,
    int32_t* out,
    int64_t n
);

bool is_leap_year(int32_t year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

int32_t days_in_month(int32_t year, int32_t month) {
    if (month == 1 || month == 3 || month == 5 || month == 7 || month == 8 || month == 10 || month == 12) {
        return 31;
    } else if (month == 4 || month == 6 || month == 9 || month == 11) {
        return 30;
    } else if (month == 2) {
        return is_leap_year(year) ? 29 : 28;
    }
    return 0;
}

void cpu_reference(const int32_t* days, int64_t n, int32_t* out_ref) {
    for (int64_t idx = 0; idx < n; idx++) {
        int32_t day_number = days[idx];

        int32_t year = 1970;
        int32_t month = 1;
        int32_t current_day = 0;

        // Fast forward to the year and month
        while (current_day + 365 < day_number) {
            int32_t days_this_year = is_leap_year(year) ? 366 : 365;
            if (current_day + days_this_year <= day_number) {
                current_day += days_this_year;
                year++;
            } else {
                break;
            }
        }

        // Find the month
        while (true) {
            int32_t days_this_month = days_in_month(year, month);
            if (current_day + days_this_month > day_number) {
                out_ref[idx] = current_day;
                break;
            }
            current_day += days_this_month;
            month++;
            if (month > 12) {
                month = 1;
                year++;
            }
        }
    }
}

void test_month_start(int64_t n) {
    // Allocate host memory
    int32_t* days_h = new int32_t[n];
    int32_t* out_h = new int32_t[n];
    int32_t* out_ref = new int32_t[n];

    // Initialize data
    for (int64_t i = 0; i < n; i++) {
        days_h[i] = (int32_t)(10 + i * 3);
    }

    memset(out_h, 0, n * sizeof(int32_t));

    // Allocate device memory
    int32_t* days_d;
    int32_t* out_d;

    cudaMalloc(&days_d, n * sizeof(int32_t));
    cudaMalloc(&out_d, n * sizeof(int32_t));

    // Copy to device
    cudaMemcpy(days_d, days_h, n * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int grid_size = (n + 255) / 256;
    if (grid_size == 0) grid_size = 1;
    kernel<<<grid_size, 256>>>(days_d, out_d, n);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(out_h, out_d, n * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    cpu_reference(days_h, n, out_ref);

    // Check results
    for (int64_t i = 0; i < n; i++) {
        assert(out_h[i] == out_ref[i]);
    }

    std::cout << "Test passed for n=" << n << std::endl;

    // Cleanup
    cudaFree(days_d);
    cudaFree(out_d);

    delete[] days_h;
    delete[] out_h;
    delete[] out_ref;
}

int main() {
    test_month_start(0);
    test_month_start(1);
    test_month_start(1000);
    test_month_start(10000);
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
