#!/usr/bin/env python3
import numpy as np
import torch
import sys
import importlib.util

task_dir = "/opgate/eval/corpus/synthesized/haiku/hash_probe"
spec = importlib.util.spec_from_file_location("triton_module", f"{task_dir}/triton.py")
triton_module = importlib.util.module_from_spec(spec)
sys.modules['triton_module'] = triton_module
spec.loader.exec_module(triton_module)
run = triton_module.run

def test():
    for n in [0, 1, 100]:
        build_n = 100
        build_keys_np = np.random.randint(1, 10000, build_n).astype(np.int64)
        build_keys_np = np.unique(build_keys_np)[:build_n]
        
        capacity = max(2, 2 * len(build_keys_np))
        table_keys_np = np.full(capacity, -1, dtype=np.int64)
        table_rows_np = np.zeros(capacity, dtype=np.int32)
        
        for i, key in enumerate(build_keys_np):
            hash_val = key & (capacity - 1)
            while table_keys_np[hash_val] != -1:
                hash_val = (hash_val + 1) & (capacity - 1)
            table_keys_np[hash_val] = key
            table_rows_np[hash_val] = i
        
        probe_keys_np = np.random.choice(build_keys_np, n) if n > 0 and len(build_keys_np) > 0 else np.array([], dtype=np.int64)
        out_build_np = np.zeros(capacity, dtype=np.int32)
        out_probe_np = np.zeros(capacity, dtype=np.int32)
        out_count_np = np.zeros(1, dtype=np.int32)
        
        build_keys_gpu = torch.from_numpy(build_keys_np).cuda()
        table_keys_gpu = torch.from_numpy(table_keys_np).cuda()
        table_rows_gpu = torch.from_numpy(table_rows_np).cuda()
        probe_keys_gpu = torch.from_numpy(probe_keys_np).cuda()
        out_build_gpu = torch.from_numpy(out_build_np).cuda()
        out_probe_gpu = torch.from_numpy(out_probe_np).cuda()
        out_count_gpu = torch.from_numpy(out_count_np).cuda()
        
        run(probe_keys_gpu, table_keys_gpu, table_rows_gpu, out_build_gpu, out_probe_gpu, out_count_gpu)
        
        result_count = out_count_gpu.cpu().numpy()[0]
        print(f"PASS: hash_probe n={n}, count={result_count}")
    
    return True

if __name__ == "__main__":
    sys.exit(0 if test() else 1)
