#include <cstdint>

extern "C" __global__ void kernel(int64_t* idx, uint8_t* l, int64_t* s, int64_t* d, int64_t* ld, int64_t* c, int64_t n){
int64_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||!l[i]||s[i]<0)return;
bool e=d[idx[i]]>=2,ne=ld[idx[i]]<=1;if(e&&ne)atomicAdd((unsigned long long*)&c[s[i]],1ULL);}
