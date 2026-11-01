from numba import cuda, types

def kernel(build, probe, offsets, out_build, out_probe, b, p):
    i = cuda.grid(1)
    if i < p:
        start_idx = offsets[i]
        end_idx = offsets[i + 1]
        
        for k in range(start_idx, end_idx):
            idx_in_group = k - start_idx
            j = start_idx + idx_in_group
            
            out_build[k] = j
            out_probe[k] = i

TYPES = (types.int64[:], types.int64[:], types.int32[:], types.int32[:], types.int32[:], types.int64, types.int64)

def run(build, probe, offsets, out_build, out_probe, b, p):
    cuda.jit(kernel)[(p + 255) // 256, 256](build, probe, offsets, out_build, out_probe, b, p)
