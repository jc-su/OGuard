#define BLOCK_SIZE 1024
#define MAX_CODE 100
__device__ bool exists_key(int64_t k, const int64_t* a, int64_t n) {
    int64_t l=0,h=n-1;
    while(l<=h){int64_t m=(l+h)>>1;int64_t v=a[m];if(v==k)return true;if(v<k)l=m+1;else h=m-1;}
    return false;
}

extern "C" __global__ void kernel(const int64_t* ck,const double* bal,const int32_t* cd,const bool* ms,const int64_t* ok,int64_t no,double av,int64_t* cnts,double* sums,int64_t n){
    int64_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && ms[i] && bal[i]>av && !exists_key(ck[i],ok,no)){
        int32_t c=cd[i];if(c>=0 && c<MAX_CODE){atomicAdd((unsigned long long*)&cnts[c],1ULL);atomicAdd(&sums[c],bal[i]);}
    }
}
