#!/usr/bin/env python3
import numpy as np
import torch
import sys
import importlib.util

task_dir = "/opgate/eval/corpus/synthesized/haiku/q6_sum"
spec = importlib.util.spec_from_file_location("triton_module", f"{task_dir}/triton.py")
triton_module = importlib.util.module_from_spec(spec)
sys.modules['triton_module'] = triton_module
spec.loader.exec_module(triton_module)
run = triton_module.run

def test():
    for n in [0, 1, 1000]:
        shipdate_np = np.random.randint(0, 1000, n).astype(np.int32)
        discount_np = np.random.rand(n).astype(np.float64) * 0.3
        quantity_np = np.random.rand(n).astype(np.float64) * 50
        price_np = np.random.rand(n).astype(np.float64) * 100
        result_np = np.zeros(1, dtype=np.float64)
        
        date_lo, date_hi = 100, 500
        disc_lo, disc_hi = 0.05, 0.2
        qty_max = 30.0
        
        mask = (shipdate_np >= date_lo) & (shipdate_np < date_hi) & \
               (discount_np >= disc_lo) & (discount_np <= disc_hi) & \
               (quantity_np < qty_max)
        expected = np.sum(price_np[mask] * discount_np[mask])
        
        shipdate_gpu = torch.from_numpy(shipdate_np).cuda()
        discount_gpu = torch.from_numpy(discount_np).cuda()
        quantity_gpu = torch.from_numpy(quantity_np).cuda()
        price_gpu = torch.from_numpy(price_np).cuda()
        result_gpu = torch.from_numpy(result_np).cuda()
        
        run(shipdate_gpu, discount_gpu, quantity_gpu, price_gpu, 
            date_lo, date_hi, disc_lo, disc_hi, qty_max, result_gpu)
        
        result = result_gpu.cpu().numpy()[0]
        if not np.isclose(result, expected, rtol=1e-9):
            print(f"FAIL: q6_sum n={n}, expected={expected}, got={result}")
            return False
        print(f"PASS: q6_sum n={n}")
    
    return True

if __name__ == "__main__":
    sys.exit(0 if test() else 1)
