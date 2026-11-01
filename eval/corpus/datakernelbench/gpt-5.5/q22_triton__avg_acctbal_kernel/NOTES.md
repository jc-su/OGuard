# q22_triton__avg_acctbal_kernel

Source: `azure__gpt-5.5/triton/full/queries/q22/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_avg_acctbal_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_avg_acctbal_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` (plan.py line 176) launches the kernel once with grid
`(triton.cdiv(n_customers, 1024),)`, `N_CODES=n_codes`,
`BLOCK_SIZE=block_size_customer` (default 1024), and no `num_warps`, so 4 warps.
`n_codes` is the count of distinct country codes of `run_query`'s default
`cntrycodes` ("13", "31", "23", "29", "30", "18", "17"), so `N_CODES` is 7.

## Lengths and ranges

- `cntrycode_ptr` (`int32`, the first two digits of `c_phone`) and `acctbal_ptr`
  (`float64`): `n_customers` rows each; `n_customers = c_custkey.numel()`, and
  `_query_core` returns before the launch when it is 0, so its least value is 1.
- `codes_ptr`: `torch.tensor(codes_int, dtype=torch.int32)`, the 7 codes.
- `out_sum_count_ptr`: `torch.zeros((2,), dtype=torch.float64)`, which the kernel
  adds to with `tl.atomic_add` at 0 and 1 (`readwrite`).
