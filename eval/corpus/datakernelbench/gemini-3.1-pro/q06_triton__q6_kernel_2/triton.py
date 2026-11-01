import triton
import triton.language as tl

TYPES = {
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "quantity_ptr": "*fp64",
    "shipdate_ptr": "*i32",
    "date_lo": "i32",
    "date_hi": "i32",
    "thresh_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
    price_ptr, discount_ptr, quantity_ptr, shipdate_ptr,
    date_lo, date_hi, thresh_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Load FP64 thresholds from memory to prevent PyTorch float -> FP32 drift
    disc_lo = tl.load(thresh_ptr)
    disc_hi = tl.load(thresh_ptr + 1)
    qty_hi = tl.load(thresh_ptr + 2)
    
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Conditional loading logic to save memory bandwidth: 
    # we progressively restrict the mask so we only load subsequent columns 
    # if the previous predicates are met.
    
    # 1. Filter on shipdate
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    keep = mask & (shipdate >= date_lo) & (shipdate < date_hi)
    
    # 2. Filter on discount
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    keep = keep & (discount >= disc_lo) & (discount <= disc_hi)
    
    # 3. Filter on quantity
    quantity = tl.load(quantity_ptr + offs, mask=keep, other=qty_hi)
    keep = keep & (quantity < qty_hi)
    
    # 4. Load price only for matches, then multiply and accumulate
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    
    tl.atomic_add(out_ptr, tl.sum(tl.where(keep, price * discount, 0.0)))
