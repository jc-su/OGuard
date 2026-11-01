import os, tempfile, torch
from torch.utils.cpp_extension import load
import cudf, pandas as pd

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#define BLOCK_SIZE 1024
__global__ void promo_kernel(const double* __restrict__ p, const double* __restrict__ d,
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
std::vector<double> promo_launcher(torch::Tensor P, torch::Tensor D, torch::Tensor SD,
    torch::Tensor PM, int32_t lo, int32_t hi){
    TORCH_CHECK(P.is_cuda()&&D.is_cuda()&&SD.is_cuda()&&PM.is_cuda());
    P=P.contiguous(); D=D.contiguous(); SD=SD.contiguous(); PM=PM.contiguous();
    auto tot=torch::zeros({1},P.options()), pr_out=torch::zeros({1},P.options());
    int64_t n=P.numel(), blocks=(n+BLOCK_SIZE-1)/BLOCK_SIZE;
    promo_kernel<<<blocks,BLOCK_SIZE>>>(P.data_ptr<double>(),D.data_ptr<double>(),
        SD.data_ptr<int32_t>(),PM.data_ptr<uint8_t>(),lo,hi,tot.data_ptr<double>(),pr_out.data_ptr<double>(),n);
    cudaDeviceSynchronize();
    return {tot.item<double>(), pr_out.item<double>()};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("promo",&promo_launcher,"promo");}
'''

def _load_cuda():
    d=tempfile.mkdtemp(); p=os.path.join(d,"k.cu"); b=os.path.join(d,"b"); os.makedirs(b,exist_ok=True)
    with open(p,"w") as f: f.write(_CUDA_SOURCE)
    return load(name="q06_ext",sources=[p],extra_cflags=["-O3"],extra_cuda_cflags=["-O3","-arch=sm_90"],with_cuda=True,build_directory=b)

_EXT = _load_cuda()

def _launch(l_ext,l_disc,l_sd,promo,lo,hi):
    for t in(l_ext,l_disc,l_sd,promo): exec(f"if not t.is_cuda: t=t.cuda()")
    total,pr = _EXT.promo(l_ext,l_disc,l_sd,promo.to(torch.uint8),lo,hi)
    return torch.tensor(total,device=l_ext.device), torch.tensor(pr,device=l_ext.device)

def _query_core(l_ext,l_disc,l_sd,promo,lo,hi):
    tot,pr = _launch(l_ext,l_disc,l_sd,promo,lo,hi)
    if tot.item()==0: return torch.tensor(0.0,device=l_ext.device)
    return 100.0*pr/tot

def run_query(tables,shipdate_lo="1995-09-01",shipdate_hi="1995-10-01"):
    df_lineitem, df_part = tables["lineitem"], tables["part"]
    lo=int(pd.to_datetime(shipdate_lo).timestamp()//86400)
    hi=int(pd.to_datetime(shipdate_hi).timestamp()//86400)
    part_proj=cudf.DataFrame({"p_partkey":df_part["p_partkey"],"p_type_is_promo":df_part["p_type"].str.startswith("PROMO")})
    j=df_lineitem.merge(part_proj,left_on="l_partkey",right_on="p_partkey")
    l_sd=torch.as_tensor(((j["l_shipdate"].astype("int64")//86400).astype("int32")).to_cupy())
    l_ext=torch.as_tensor(j["l_extendedprice"].to_cupy(),dtype=torch.float64)
    l_disc=torch.as_tensor(j["l_discount"].to_cupy(),dtype=torch.float64)
    promo=torch.as_tensor(j["p_type_is_promo"].astype("bool").to_cupy())
    return float(_query_core(l_ext,l_disc,l_sd,promo,lo,hi).item())
