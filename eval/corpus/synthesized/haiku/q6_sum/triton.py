import torch
import triton
import triton.language as tl

TYPES = {
    "shipdate": "*i32",
    "discount": "*fp64",
    "quantity": "*fp64",
    "price": "*fp64",
    "result": "*fp64",
    "date_lo": "i32",
    "date_hi": "i32",
    "disc_lo": "fp64",
    "disc_hi": "fp64",
    "qty_max": "fp64",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4

@triton.jit
def kernel(shipdate, discount, quantity, price, date_lo, date_hi, disc_lo, disc_hi, qty_max, result, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    ship_vals = tl.load(shipdate + offsets, mask=mask, other=0)
    disc_vals = tl.load(discount + offsets, mask=mask, other=0.0)
    qty_vals = tl.load(quantity + offsets, mask=mask, other=0.0)
    price_vals = tl.load(price + offsets, mask=mask, other=0.0)
    
    is_valid = (ship_vals >= date_lo) & (ship_vals < date_hi) & (disc_vals >= disc_lo) & (disc_vals <= disc_hi) & (qty_vals < qty_max)
    contribution = tl.where(is_valid & mask, price_vals * disc_vals, 0.0)
    
    sum_val = tl.sum(contribution)
    tl.atomic_add(result, 0, sum_val)

def run(shipdate: torch.Tensor, discount: torch.Tensor, quantity: torch.Tensor, price: torch.Tensor, 
        date_lo: int, date_hi: int, disc_lo: float, disc_hi: float, qty_max: float, result: torch.Tensor) -> None:
    n = shipdate.numel()
    BLOCK_SIZE = CONSTANTS["BLOCK_SIZE"]
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    kernel[grid](shipdate, discount, quantity, price, date_lo, date_hi, disc_lo, disc_hi, qty_max, result, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
