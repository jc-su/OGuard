import os
import tempfile
import torch
from torch.utils.cpp_extension import load
import cudf
from typing import Tuple

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#define BLOCK_SIZE 1024
#define MAX_CODE 100
__device__ bool exists_key(int64_t k, const int64_t* a, int64_t n) {
    int64_t l=0,h=n-1;
    while(l<=h){int64_t m=(l+h)>>1;int64_t v=a[m];if(v==k)return true;if(v<k)l=m+1;else h=m-1;}
    return false;
}
__global__ void ker_avg(const double* b,const bool* m,double* s,int64_t* c,int64_t n){
    __shared__ double ss[BLOCK_SIZE];__shared__ int64_t cc[BLOCK_SIZE];
    int64_t i=blockIdx.x*blockDim.x+threadIdx.x;double ls=0;int64_t lc=0;
    if(i<n && m[i] && b[i]>0.0){ls=b[i];lc=1;}
    ss[threadIdx.x]=ls;cc[threadIdx.x]=lc;__syncthreads();
    for(int s_=BLOCK_SIZE/2;s_>0;s_>>=1){if(threadIdx.x<s_){ss[threadIdx.x]+=ss[threadIdx.x+s_];cc[threadIdx.x]+=cc[threadIdx.x+s_];}__syncthreads();}
    if(threadIdx.x==0){atomicAdd(s,ss[0]);atomicAdd((unsigned long long*)c,(unsigned long long)cc[0]);}
}
__global__ void ker_grp(const int64_t* ck,const double* bal,const int32_t* cd,const bool* ms,const int64_t* ok,int64_t no,double av,int64_t* cnts,double* sums,int64_t n){
    int64_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && ms[i] && bal[i]>av && !exists_key(ck[i],ok,no)){
        int32_t c=cd[i];if(c>=0 && c<MAX_CODE){atomicAdd((unsigned long long*)&cnts[c],1ULL);atomicAdd(&sums[c],bal[i]);}
    }
}
std::tuple<torch::Tensor,torch::Tensor> cuda_avg(torch::Tensor a,torch::Tensor m){
    a=a.contiguous();m=m.contiguous();auto s=torch::zeros({1},a.options());auto c=torch::zeros({1},a.options().dtype(torch::kInt64));
    ker_avg<<<(a.numel()+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(a.data_ptr<double>(),m.data_ptr<bool>(),s.data_ptr<double>(),c.data_ptr<int64_t>(),a.numel());
    return {s,c};
}
std::tuple<torch::Tensor,torch::Tensor> cuda_grp(torch::Tensor ck,torch::Tensor bal,torch::Tensor cd,torch::Tensor ms,torch::Tensor ok,double av){
    ck=ck.contiguous();bal=bal.contiguous();cd=cd.contiguous();ms=ms.contiguous();ok=ok.contiguous();
    auto cnts=torch::zeros({MAX_CODE},ck.options().dtype(torch::kInt64));auto sums=torch::zeros({MAX_CODE},bal.options());
    ker_grp<<<(ck.numel()+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(ck.data_ptr<int64_t>(),bal.data_ptr<double>(),cd.data_ptr<int32_t>(),ms.data_ptr<bool>(),ok.data_ptr<int64_t>(),ok.numel(),av,cnts.data_ptr<int64_t>(),sums.data_ptr<double>(),ck.numel());
    return {cnts,sums};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("cuda_avg",&cuda_avg);m.def("cuda_grp",&cuda_grp);}
'''

def _load_cuda():
    d=tempfile.mkdtemp(prefix="q22_");p=os.path.join(d,"k.cu");os.makedirs(os.path.join(d,"b"),exist_ok=True)
    with open(p,"w") as f:f.write(_CUDA_SOURCE)
    return load("q22_ext",sources=[p],extra_cflags=["-O3"],extra_cuda_cflags=["-O3","-arch=sm_90"],with_cuda=True,build_directory=os.path.join(d,"b"))

_EXT=_load_cuda()

def _query_core(c_custkey:torch.Tensor,c_acctbal:torch.Tensor,cntrycode:torch.Tensor,in_mask:torch.Tensor,o_custkey_unique:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    s_ord,_=torch.sort(o_custkey_unique)
    s,c=_EXT.cuda_avg(c_acctbal,in_mask)
    avg_val=s.item()/c.item() if c.item()>0 else 0.0
    cnts,sums=_EXT.cuda_grp(c_custkey,c_acctbal,cntrycode,in_mask,s_ord,avg_val)
    valid=cnts>0
    codes=torch.nonzero(valid,as_tuple=True)[0].int()
    nc=cnts[valid].int()
    tb=sums[valid]
    order=torch.argsort(codes)
    return codes[order].to(torch.int32),nc[order],tb[order]

def run_query(tables:dict[str,cudf.DataFrame],cntrycodes:tuple=('13','31','23','29','30','18','17'))->cudf.DataFrame:
    df_c=tables["customer"];df_o=tables["orders"]
    cntry_series=df_c["c_phone"].str.slice(0,2).astype("int32")
    codes_int=[int(x) for x in cntrycodes]
    in_mask_series=cntry_series.isin(codes_int)
    o_unique=df_o["o_custkey"].unique()
    c_custkey_t=torch.as_tensor(df_c["c_custkey"].to_cupy(),dtype=torch.int64)
    c_acctbal_t=torch.as_tensor(df_c["c_acctbal"].to_cupy(),dtype=torch.float64)
    cntrycode_t=torch.as_tensor(cntry_series.to_cupy(),dtype=torch.int32)
    in_mask_t=torch.as_tensor(in_mask_series.to_cupy(),dtype=torch.bool)
    o_custkey_t=torch.as_tensor(o_unique.to_cupy(),dtype=torch.int64)
    codes_t,numcust_t,totacctbal_t=_query_core(c_custkey_t,c_acctbal_t,cntrycode_t,in_mask_t,o_custkey_t)
    codes_cudf=cudf.Series(codes_t).astype("int32").astype("str").str.zfill(2)
    return cudf.DataFrame({"cntrycode":codes_cudf,"numcust":cudf.Series(numcust_t),"totacctbal":cudf.Series(totacctbal_t)})
