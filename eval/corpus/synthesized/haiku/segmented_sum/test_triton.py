#!/usr/bin/env python3
import numpy as np
import torch
import sys
import importlib.util

task_dir = "/opgate/eval/corpus/synthesized/haiku/segmented_sum"
spec = importlib.util.spec_from_file_location("triton_module", f"{task_dir}/triton.py")
triton_module = importlib.util.module_from_spec(spec)
sys.modules['triton_module'] = triton_module
spec.loader.exec_module(triton_module)
run = triton_module.run

def test():
    for n in [0, 1, 1000]:
        if "segmented_sum" == "gather":
            m = n
            col_np = np.random.rand(n).astype(np.float64)
            sel_np = (np.random.rand(m) * max(1, n)).astype(np.int32) if m > 0 else np.array([], dtype=np.int32)
            out_np = np.zeros(m, dtype=np.float64)
            
            if m == 0:
                expected = np.array([], dtype=np.float64)
            else:
                sel_np = np.minimum(sel_np, n-1) if n > 0 else sel_np
                expected = col_np[sel_np] if n > 0 else out_np
            
            col_gpu = torch.from_numpy(col_np).cuda()
            sel_gpu = torch.from_numpy(sel_np).cuda()
            out_gpu = torch.from_numpy(out_np).cuda()
            
            run(col_gpu, sel_gpu, out_gpu)
            result = out_gpu.cpu().numpy()
            
            if m > 0 and not np.allclose(result, expected, rtol=1e-9, atol=1e-14):
                print(f"FAIL: gather n={n}")
                return False
            print(f"PASS: gather n={n}")
        
        elif "segmented_sum" == "histogram":
            keys_np = np.random.randint(0, 256, n, dtype=np.uint32)
            counts_np = np.zeros(256, dtype=np.int32)
            shift = 0
            
            expected = np.bincount((keys_np >> shift) & 255, minlength=256).astype(np.int32)
            
            keys_gpu = torch.from_numpy(keys_np).cuda()
            counts_gpu = torch.from_numpy(counts_np).cuda()
            
            run(keys_gpu, shift, counts_gpu)
            result = counts_gpu.cpu().numpy()
            
            if not np.array_equal(result, expected):
                print(f"FAIL: histogram n={n}")
                return False
            print(f"PASS: histogram n={n}")
        
        elif "segmented_sum" == "segmented_sum":
            g = max(1, n // 100) if n > 0 else 1
            values_np = np.random.rand(n).astype(np.float64)
            if g < n:
                offsets_np = np.sort(np.random.choice(n, g, replace=False)).astype(np.int32)
            else:
                offsets_np = np.array([], dtype=np.int32)
            offsets_np = np.concatenate([offsets_np, [n]])
            offsets_np = np.unique(offsets_np)
            g = len(offsets_np) - 1
            sums_np = np.zeros(g, dtype=np.float64)
            
            expected = np.array([np.sum(values_np[offsets_np[i]:offsets_np[i+1]]) for i in range(g)])
            
            values_gpu = torch.from_numpy(values_np).cuda()
            offsets_gpu = torch.from_numpy(offsets_np).cuda()
            sums_gpu = torch.from_numpy(sums_np).cuda()
            
            run(offsets_gpu, values_gpu, sums_gpu)
            result = sums_gpu.cpu().numpy()
            
            if not np.allclose(result, expected, rtol=1e-9, atol=1e-14):
                print(f"FAIL: segmented_sum n={n}, g={g}")
                return False
            print(f"PASS: segmented_sum n={n}, g={g}")
    
    return True

if __name__ == "__main__":
    sys.exit(0 if test() else 1)
