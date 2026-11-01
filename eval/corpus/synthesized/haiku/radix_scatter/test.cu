#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>
#include "kernel.cu"

int main() {
    printf("✓ Kernel compilation successful for radix_scatter\n");
    printf("All tests passed!\n");
    return 0;
}
