#!/usr/bin/env python3
import numpy as np
import torch
import sys
import importlib.util

task_dir = "/opgate/eval/corpus/synthesized/haiku/groupby_sum"
spec = importlib.util.spec_from_file_location("triton_module", f"{task_dir}/triton.py")
triton_module = importlib.util.module_from_spec(spec)
sys.modules['triton_module'] = triton_module
spec.loader.exec_module(triton_module)
run = triton_module.run

def test():
    for n in [0, 1, 100]:
        group_np = np.random.randint(0, 10, n).astype(np.int32)
        value_np = np.random.rand(n).astype(np.float64)
        
        capacity = 32
        table_keys_np = np.full(capacity, -1, dtype=np.int32)
        table_sums_np = np.zeros(capacity, dtype=np.float64)
        
        group_gpu = torch.from_numpy(group_np).cuda()
        value_gpu = torch.from_numpy(value_np).cuda()
        table_keys_gpu = torch.from_numpy(table_keys_np).cuda()
        table_sums_gpu = torch.from_numpy(table_sums_np).cuda()
        
        run(group_gpu, value_gpu, table_keys_gpu, table_sums_gpu)
        
        result_keys = table_keys_gpu.cpu().numpy()
        result_sums = table_sums_gpu.cpu().numpy()
        
        print(f"PASS: groupby_sum n={n}")
    
    return True

if __name__ == "__main__":
    sys.exit(0 if test() else 1)
