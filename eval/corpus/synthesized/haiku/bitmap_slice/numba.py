from numba import cuda, types

def kernel(source, out, n, offset):
    i = cuda.grid(1)
    if i < n:
        src_byte_idx = (offset + i) // 8
        src_bit_idx = (offset + i) % 8
        src_bit = (source[src_byte_idx] >> src_bit_idx) & 1
        
        out_byte_idx = i // 8
        out_bit_idx = i % 8
        
        if src_bit:
            out[out_byte_idx] = out[out_byte_idx] | (1 << out_bit_idx)
        else:
            out[out_byte_idx] = out[out_byte_idx] & ~(1 << out_bit_idx)

TYPES = (types.uint8[:], types.uint8[:], types.int64, types.int64)

def run(source, out, n, offset):
    out[:] = 0
    cuda.jit(kernel)[(n + 255) // 256, 256](source, out, n, offset)
