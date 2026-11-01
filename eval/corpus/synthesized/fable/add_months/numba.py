from numba import cuda, int32, int64


def kernel(days, months, out, n):
    # One thread per row: civil date from days (Hinnant's algorithm, all
    # intermediate values nonnegative for years 1 to 9999), month arithmetic,
    # day clamped to the target month, and back to days.
    i = int64(cuda.blockIdx.x) * int64(cuda.blockDim.x) + int64(cuda.threadIdx.x)
    if i < n:
        z = int64(days[i]) + 719468
        era = z // 146097
        doe = z - era * 146097
        yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
        doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
        mp = (5 * doy + 2) // 153
        d = doy - (153 * mp + 2) // 5 + 1
        m = mp + 3 if mp < 10 else mp - 9
        y = yoe + era * 400
        if m <= 2:
            y += 1
        total = y * 12 + (m - 1) + int64(months[i])
        ty = total // 12
        tm = total - ty * 12 + 1
        if tm == 2:
            leap = (ty % 4 == 0 and ty % 100 != 0) or ty % 400 == 0
            dim = 29 if leap else 28
        else:
            dim = 31 - ((tm - 1) % 7) % 2
        if d > dim:
            d = dim
        yy = ty - 1 if tm <= 2 else ty
        era2 = yy // 400
        yoe2 = yy - era2 * 400
        mp2 = tm - 3 if tm > 2 else tm + 9
        doy2 = (153 * mp2 + 2) // 5 + d - 1
        doe2 = yoe2 * 365 + yoe2 // 4 - yoe2 // 100 + doy2
        out[i] = int32(era2 * 146097 + doe2 - 719468)


TYPES = (int32[:], int32[:], int32[:], int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(days, months, out, n):
    blocks = max(1, (n + 255) // 256)
    _jit()[blocks, 256](days, months, out, n)
