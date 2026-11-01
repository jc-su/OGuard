from numba import cuda, types

def kernel(a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n):
    i = cuda.grid(1)
    if i < n:
        a_start = a_offsets[i]
        a_end = a_offsets[i + 1]
        b_start = b_offsets[i]
        b_end = b_offsets[i + 1]
        
        out_start = out_offsets[i]
        
        a_len = a_end - a_start
        b_len = b_end - b_start
        
        for j in range(a_len):
            out_chars[out_start + j] = a_chars[a_start + j]
        
        for j in range(b_len):
            out_chars[out_start + a_len + j] = b_chars[b_start + j]

TYPES = (types.int32[:], types.uint8[:], types.int32[:], types.uint8[:], 
         types.int32[:], types.uint8[:], types.int64)

def run(a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n):
    cuda.jit(kernel)[(n + 255) // 256, 256](a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n)
