#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(const double* __restrict__ p, const double* __restrict__ d,
    const int32_t* __restrict__ sd, const uint8_t* __restrict__ pm, int32_t lo, int32_t hi,
    double* __restrict__ tot, double* __restrict__ pr, int64_t n) {
    __shared__ double b_tot[BLOCK_SIZE], b_pr[BLOCK_SIZE];
    int64_t i = blockIdx.x*blockDim.x + threadIdx.x;
    double my_t=0, my_p=0;
    if(i<n && sd[i]>=lo && sd[i]<hi) { double r=p[i]*(1.0-d[i]); my_t=r; if(pm[i]) my_p=r; }
    b_tot[threadIdx.x]=my_t; b_pr[threadIdx.x]=my_p; __syncthreads();
    for(int s=blockDim.x/2; s>0; s>>=1){ if(threadIdx.x<s){b_tot[threadIdx.x]+=b_tot[threadIdx.x+s]; b_pr[threadIdx.x]+=b_pr[threadIdx.x+s];} __syncthreads();}
    if(threadIdx.x==0){ atomicAdd(tot,b_tot[0]); atomicAdd(pr,b_pr[0]); }
}
