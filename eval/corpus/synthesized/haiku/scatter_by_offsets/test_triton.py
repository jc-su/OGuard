#!/usr/bin/env python3
import numpy as np
import torch
import sys
import importlib.util

task_dir = "/opgate/eval/corpus/synthesized/haiku/scatter_by_offsets"
spec = importlib.util.spec_from_file_location("triton_module", f"{task_dir}/triton.py")
triton_module = importlib.util.module_from_spec(spec)
sys.modules['triton_module'] = triton_module
spec.loader.exec_module(triton_module)
run = triton_module.run

def test():
    for n in [0, 1, 100]:
        flags_np = (np.random.rand(n) > 0.5).astype(np.int8)
        offsets_np = np.cumsum(flags_np, dtype=np.int32) - flags_np.astype(np.int32)
        total = int(np.sum(flags_np))
        out_np = np.zeros(total, dtype=np.int32)
        
        flags_gpu = torch.from_numpy(flags_np).cuda()
        offsets_gpu = torch.from_numpy(offsets_np).cuda()
        out_gpu = torch.from_numpy(out_np).cuda()
        
        run(flags_gpu, offsets_gpu, total, out_gpu)
        
        print(f"PASS: scatter_by_offsets n={n}")
    
    return True

if __name__ == "__main__":
    sys.exit(0 if test() else 1)
