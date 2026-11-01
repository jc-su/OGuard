from numba import cuda, types

def kernel(group, value, best, n, g):
    i = cuda.grid(1)
    if i < n:
        grp = group[i]
        val = value[i]
        
        old_best = best[grp]
        old_best_val = -9223372036854775808 if old_best == -1 else value[old_best]
        
        if val > old_best_val:
            cuda.atomic.min(best, grp, i)

TYPES = (types.int32[:], types.int64[:], types.int32[:], types.int64, types.int64)

def run(group, value, best, n, g):
    cuda.jit(kernel)[(n + 255) // 256, 256](group, value, best, n, g)
