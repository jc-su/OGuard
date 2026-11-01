import os, tempfile, torch, cudf
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#define B 1024
__global__ void k(int64_t* idx, uint8_t* l, int64_t* s, int64_t* d, int64_t* ld, int64_t* c, int64_t n){
int64_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||!l[i]||s[i]<0)return;
bool e=d[idx[i]]>=2,ne=ld[idx[i]]<=1;if(e&&ne)atomicAdd((unsigned long long*)&c[s[i]],1ULL);}
void launch(torch::Tensor I,torch::Tensor L,torch::Tensor S,torch::Tensor D,torch::Tensor LD,torch::Tensor C,int64_t n){
TORCH_CHECK(I.is_cuda());auto I_=I.contiguous(),L_=L.contiguous(),S_=S.contiguous(),D_=D.contiguous(),LD_=LD.contiguous(),C_=C.contiguous();
int64_t g=(n+B-1)/B;k<<<g,B>>>(I_.data_ptr<int64_t>(),L_.data_ptr<uint8_t>(),S_.data_ptr<int64_t>(),D_.data_ptr<int64_t>(),LD_.data_ptr<int64_t>(),C_.data_ptr<int64_t>(),n);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("launch",&launch,"q21");}'''

def _load_cuda():
    d=tempfile.mkdtemp();p=os.path.join(d,"k.cu");b=os.path.join(d,"b");os.makedirs(b,exist_ok=True)
    with open(p,"w")as f:f.write(_CUDA_SOURCE)
    return load(name="q21_cuda",sources=[p],extra_cflags=["-O3"],extra_cuda_cflags=["-O3","-arch=sm_90"],with_cuda=True,build_directory=b)
_E=_load_cuda()

def _query_core(l1_orderkey,l1_suppkey,l1_late,supp_ids,lo_orderkeys,lo_distinct_supp,lo_late_distinct_supp,num_suppliers):
    idx=torch.searchsorted(lo_orderkeys,l1_orderkey)
    l1_late_u8=l1_late.to(torch.uint8)
    counts=torch.zeros(num_suppliers,dtype=torch.int64,device=l1_orderkey.device)
    n=int(idx.numel())
    _E.launch(idx,l1_late_u8,supp_ids,lo_distinct_supp,lo_late_distinct_supp,counts,n)
    return torch.arange(num_suppliers,device=counts.device,dtype=torch.int64),counts

def run_query(tables,nation_name="SAUDI ARABIA",orderstatus="F"):
    df_supplier=tables["supplier"];df_lineitem=tables["lineitem"];df_orders=tables["orders"];df_nation=tables["nation"]
    nation_f=df_nation[df_nation["n_name"]==nation_name][["n_nationkey"]]
    supp_n=df_supplier.merge(nation_f,left_on="s_nationkey",right_on="n_nationkey")[["s_suppkey","s_name"]].sort_values("s_name").reset_index(drop=True)
    supp_n["supp_id"]=cudf.Series(range(len(supp_n)),dtype="int32")
    orders_f=df_orders[df_orders["o_orderstatus"]==orderstatus][["o_orderkey"]]
    li_all=df_lineitem[["l_orderkey","l_suppkey","l_receiptdate","l_commitdate"]].merge(orders_f,left_on="l_orderkey",right_on="o_orderkey")
    li_all["late"]=(li_all["l_receiptdate"]>li_all["l_commitdate"]).astype("int8")
    distinct_supp=li_all[["l_orderkey","l_suppkey"]].drop_duplicates().groupby("l_orderkey").size().reset_index()
    distinct_supp.columns=["l_orderkey","distinct_supp"]
    late_rows=li_all[li_all["late"]==1][["l_orderkey","l_suppkey"]].drop_duplicates()
    late_distinct=late_rows.groupby("l_orderkey").size().reset_index()
    late_distinct.columns=["l_orderkey","late_distinct_supp"]
    agg=distinct_supp.merge(late_distinct,on="l_orderkey",how="left")
    agg["late_distinct_supp"]=agg["late_distinct_supp"].fillna(0).astype("int32")
    agg["distinct_supp"]=agg["distinct_supp"].astype("int32")
    agg=agg.sort_values("l_orderkey").reset_index(drop=True)
    l1=df_lineitem[["l_orderkey","l_suppkey","l_receiptdate","l_commitdate"]].merge(supp_n,left_on="l_suppkey",right_on="s_suppkey").merge(orders_f,left_on="l_orderkey",right_on="o_orderkey")
    l1["late"]=(l1["l_receiptdate"]>l1["l_commitdate"]).astype("bool")
    l1_orderkey_t=torch.as_tensor(l1["l_orderkey"].to_cupy(),dtype=torch.int64)
    l1_suppkey_t=torch.as_tensor(l1["l_suppkey"].to_cupy(),dtype=torch.int64)
    l1_late_t=torch.as_tensor(l1["late"].to_cupy(),dtype=torch.bool)
    supp_ids_t=torch.as_tensor(l1["supp_id"].to_cupy(),dtype=torch.int64)
    lo_orderkeys_t=torch.as_tensor(agg["l_orderkey"].to_cupy(),dtype=torch.int64)
    lo_distinct_supp_t=torch.as_tensor(agg["distinct_supp"].to_cupy(),dtype=torch.int64)
    lo_late_distinct_supp_t=torch.as_tensor(agg["late_distinct_supp"].to_cupy(),dtype=torch.int64)
    num_suppliers=len(supp_n)
    supp_idx_t,counts_t=_query_core(l1_orderkey_t,l1_suppkey_t,l1_late_t,supp_ids_t,lo_orderkeys_t,lo_distinct_supp_t,lo_late_distinct_supp_t,num_suppliers)
    result=cudf.DataFrame({"s_name":supp_n["s_name"].iloc[cudf.Series(supp_idx_t)].reset_index(drop=True),"numwait":cudf.Series(counts_t)})
    result=result[result["numwait"]>0].sort_values(["numwait","s_name"],ascending=[False,True]).head(100).reset_index(drop=True)
    return result
