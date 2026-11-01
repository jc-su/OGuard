#!/usr/bin/env python3
import numpy as np
import torch
import sys
import importlib.util

task_dir = "/opgate/eval/corpus/synthesized/haiku/starts_with"
spec = importlib.util.spec_from_file_location("triton_module", f"{task_dir}/triton.py")
triton_module = importlib.util.module_from_spec(spec)
sys.modules['triton_module'] = triton_module
spec.loader.exec_module(triton_module)
run = triton_module.run

def test():
    for n_strings in [0, 1, 10]:
        if n_strings == 0:
            offsets_np = np.array([0], dtype=np.int32)
            chars_np = np.array([], dtype=np.uint8)
        else:
            offsets_np = np.array([0, 5, 10], dtype=np.int32)[:n_strings+1]
            chars_np = np.array([72, 105, 33, 72, 105], dtype=np.uint8)
        
        n = len(offsets_np) - 1
        prefix_np = np.array([72], dtype=np.uint8)
        match_np = np.zeros(n, dtype=np.int8)
        
        offsets_gpu = torch.from_numpy(offsets_np).cuda()
        chars_gpu = torch.from_numpy(chars_np).cuda()
        prefix_gpu = torch.from_numpy(prefix_np).cuda()
        match_gpu = torch.from_numpy(match_np).cuda()
        
        run(offsets_gpu, chars_gpu, prefix_gpu, match_gpu)
        
        print(f"PASS: starts_with n={n}")
    
    return True

if __name__ == "__main__":
    sys.exit(0 if test() else 1)
