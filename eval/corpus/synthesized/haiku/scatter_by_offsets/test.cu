#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void kernel(...);

int main() {
    printf("Test placeholder for scatter_by_offsets\n");
    return 0;
}
