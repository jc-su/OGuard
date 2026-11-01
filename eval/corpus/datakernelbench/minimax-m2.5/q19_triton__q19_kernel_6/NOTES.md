# q19_triton__q19_kernel_6

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/full/queries/q19/round_05.py`, copied as `plan.py`.

## Extraction

- The kernel `_q19_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 6 of 8, `BLOCK_SIZE = 4096` and `num_warps = 8`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`run_query` launches `_K._q19_kernel[grid]` with keyword arguments and `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `l_quantity_ptr`: `n_elements` elements: one element per joined row.
- `l_extendedprice_ptr`: `n_elements` elements: one element per joined row.
- `l_discount_ptr`: `n_elements` elements: one element per joined row.
- `p_brand_id_ptr`: `n_elements` elements: one element per joined row.
- `p_container_grp_ptr`: `n_elements` elements: one element per joined row.
- `p_size_ptr`: `n_elements` elements: one element per joined row.
- `brand12_id`: the plan passes the constant 12.
- `brand23_id`: the plan passes the constant 23.
- `brand34_id`: the plan passes the constant 34.
- `sm_grp`: the plan passes the constant 1.
- `md_grp`: the plan passes the constant 2.
- `lg_grp`: the plan passes the constant 3.
- `q1_lo`: the constant 1.0, passed as fp32; not used in an address.
- `q1_hi`: the constant 11.0, passed as fp32; not used in an address.
- `q2_lo`: the constant 10.0, passed as fp32; not used in an address.
- `q2_hi`: the constant 20.0, passed as fp32; not used in an address.
- `q3_lo`: the constant 20.0, passed as fp32; not used in an address.
- `q3_hi`: the constant 30.0, passed as fp32; not used in an address.
- `s1_lo`: the plan passes the constant 1.
- `s1_hi`: the plan passes the constant 5.
- `s2_lo`: the plan passes the constant 1.
- `s2_hi`: the plan passes the constant 10.
- `s3_lo`: the plan passes the constant 1.
- `s3_hi`: the plan passes the constant 15.
- `out_ptr`: `torch.zeros(1)`; the kernel adds to it with `tl.atomic_add`.
- `n_elements`: `n = l_quantity_t.numel()`, the size symbol of the row buffers.
