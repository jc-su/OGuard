#!/usr/bin/env python3
import numpy as np
import torch
import sys
import importlib.util

task_dir = "/opgate/eval/corpus/synthesized/haiku/filter_select"
spec = importlib.util.spec_from_file_location("triton_module", f"{task_dir}/triton.py")
triton_module = importlib.util.module_from_spec(spec)
sys.modules['triton_module'] = triton_module
spec.loader.exec_module(triton_module)
run = triton_module.run

def test():
    for n in [0, 1, 100]:
        quantity_np = np.random.rand(n).astype(np.float64)
        threshold = 0.5
        sel_np = np.zeros(n, dtype=np.int32)
        count_np = np.zeros(1, dtype=np.int32)
        
        expected_indices = np.where(quantity_np < threshold)[0]
        expected_count = len(expected_indices)
        
        quantity_gpu = torch.from_numpy(quantity_np).cuda()
        sel_gpu = torch.from_numpy(sel_np).cuda()
        count_gpu = torch.from_numpy(count_np).cuda()
        
        run(quantity_gpu, threshold, sel_gpu, count_gpu)
        
        result_count = count_gpu.cpu().numpy()[0]
        if result_count != expected_count:
            print(f"FAIL: filter_select n={n}, count={result_count}, expected={expected_count}")
            return False
        print(f"PASS: filter_select n={n}, count={result_count}")
    
    return True

if __name__ == "__main__":
    sys.exit(0 if test() else 1)
