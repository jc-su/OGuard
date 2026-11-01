from numba import cuda, types

def kernel(offsets, chars, out, valid, n):
    i = cuda.grid(1)
    if i < n:
        start = offsets[i]
        end = offsets[i + 1]
        
        out[i] = 0
        
        pos = start
        if pos < end and chars[pos] == ord('+') or chars[pos] == ord('-'):
            pos += 1
        
        if pos >= end:
            return
        
        negative = chars[start] == ord('-') if start < end and (chars[start] == ord('-') or chars[start] == ord('+')) else False
        
        result = 0
        for j in range(start + (1 if start < end and (chars[start] == ord('+') or chars[start] == ord('-')) else 0), end):
            c = chars[j]
            if c < ord('0') or c > ord('9'):
                return
            result = result * 10 + (c - ord('0'))
        
        out[i] = -result if negative else result
        
        byte_idx = i // 8
        bit_idx = i % 8
        valid[byte_idx] |= (1 << bit_idx)

TYPES = (types.int32[:], types.uint8[:], types.int64[:], types.uint8[:], types.int64)

def run(offsets, chars, out, valid, n):
    cuda.jit(kernel)[(n + 255) // 256, 256](offsets, chars, out, valid, n)
