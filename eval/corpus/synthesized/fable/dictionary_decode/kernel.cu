// dictionary_decode: gather dictionary values through int32 codes.
//
// One thread per row: out[i] = dictionary[codes[i]]. Code loads and output
// stores are coalesced; the dictionary gather is served by L1/L2.

extern "C" __global__ void kernel(const int* __restrict__ codes,
                                  const double* __restrict__ dictionary,
                                  double* __restrict__ out,
                                  long long n,
                                  long long d)
{
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i < n) out[i] = dictionary[codes[i]];
}
