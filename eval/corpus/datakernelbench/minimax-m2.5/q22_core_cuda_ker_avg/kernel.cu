#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(const double* b,const bool* m,double* s,int64_t* c,int64_t n){
    __shared__ double ss[BLOCK_SIZE];__shared__ int64_t cc[BLOCK_SIZE];
    int64_t i=blockIdx.x*blockDim.x+threadIdx.x;double ls=0;int64_t lc=0;
    if(i<n && m[i] && b[i]>0.0){ls=b[i];lc=1;}
    ss[threadIdx.x]=ls;cc[threadIdx.x]=lc;__syncthreads();
    for(int s_=BLOCK_SIZE/2;s_>0;s_>>=1){if(threadIdx.x<s_){ss[threadIdx.x]+=ss[threadIdx.x+s_];cc[threadIdx.x]+=cc[threadIdx.x+s_];}__syncthreads();}
    if(threadIdx.x==0){atomicAdd(s,ss[0]);atomicAdd((unsigned long long*)c,(unsigned long long)cc[0]);}
}
