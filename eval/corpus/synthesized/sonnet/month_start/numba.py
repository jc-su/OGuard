from numba import cuda, int32, int64

TYPES = (int32[:], int32[:], int64)


def kernel(days, out, n):
    # Civil-from-days (proleptic Gregorian, March-based years). The month start
    # is the day number minus (day of month - 1). Inputs are >= -719162, so
    # z below is positive and integer division is exact.
    stride = cuda.gridDim.x * cuda.blockDim.x
    i = cuda.grid(1)
    while i < n:
        day = days[i]
        z = day + 719468
        era = z // 146097
        doe = z - era * 146097
        yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
        doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
        mp = (5 * doy + 2) // 153
        dom = doy - (153 * mp + 2) // 5 + 1
        out[i] = day - (dom - 1)
        i += stride


def run(days, out, n):
    cuda.jit(kernel)[max(1, (n + 255) // 256), 256](days, out, n)
