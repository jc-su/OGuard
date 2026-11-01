#!/usr/bin/env python3
import numpy as np
import torch
import sys
import os
import importlib.util

task_dir = "/opgate/eval/corpus/synthesized/haiku/projection"
spec = importlib.util.spec_from_file_location("triton_module", f"{task_dir}/triton.py")
triton_module = importlib.util.module_from_spec(spec)
sys.modules['triton_module'] = triton_module
spec.loader.exec_module(triton_module)
run = triton_module.run

def test():
    for n in [0, 1, 1000, 1000003]:
        price_np = np.random.rand(n).astype(np.float64)
        discount_np = np.random.rand(n).astype(np.float64)
        out_np = np.zeros(n, dtype=np.float64)
        
        expected = price_np * (1.0 - discount_np)
        
        price_gpu = torch.from_numpy(price_np).cuda()
        discount_gpu = torch.from_numpy(discount_np).cuda()
        out_gpu = torch.from_numpy(out_np).cuda()
        
        run(price_gpu, discount_gpu, out_gpu)
        result = out_gpu.cpu().numpy()
        
        if not np.allclose(result, expected, rtol=1e-9):
            print(f"FAIL: n={n}")
            return False
        print(f"PASS: n={n}")
    return True

if __name__ == "__main__":
    sys.exit(0 if test() else 1)
