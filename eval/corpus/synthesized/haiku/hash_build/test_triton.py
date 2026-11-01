#!/usr/bin/env python3
import numpy as np
import torch
import sys
import importlib.util

task_dir = "/opgate/eval/corpus/synthesized/haiku/hash_build"
spec = importlib.util.spec_from_file_location("triton_module", f"{task_dir}/triton.py")
triton_module = importlib.util.module_from_spec(spec)
sys.modules['triton_module'] = triton_module
spec.loader.exec_module(triton_module)
run = triton_module.run

def test():
    for n in [0, 1, 100]:
        if n == 0:
            keys_np = np.array([], dtype=np.int64)
            capacity = 2
        else:
            keys_np = np.random.randint(1, 1000000, n).astype(np.int64)
            keys_np = np.unique(keys_np)[:n]
            capacity = max(2, 2 * len(keys_np))
        
        table_keys_np = np.full(capacity, -1, dtype=np.int64)
        table_rows_np = np.zeros(capacity, dtype=np.int32)
        
        keys_gpu = torch.from_numpy(keys_np).cuda()
        table_keys_gpu = torch.from_numpy(table_keys_np).cuda()
        table_rows_gpu = torch.from_numpy(table_rows_np).cuda()
        
        run(keys_gpu, table_keys_gpu, table_rows_gpu)
        
        result_keys = table_keys_gpu.cpu().numpy()
        for key in keys_np:
            if key not in result_keys:
                print(f"FAIL: hash_build n={n}, key {key} not found")
                return False
        print(f"PASS: hash_build n={n}")
    
    return True

if __name__ == "__main__":
    sys.exit(0 if test() else 1)
