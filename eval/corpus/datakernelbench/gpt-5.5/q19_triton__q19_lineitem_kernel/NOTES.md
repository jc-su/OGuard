# q19_triton__q19_lineitem_kernel

Source: `azure__gpt-5.5/triton/full/queries/q19/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q19_lineitem_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q19_lineitem_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` (plan.py line 282) launches the kernel once with grid
`(triton.cdiv(n, 2048),)`, `BLOCK_SIZE=block_size` (2048 from `run_query`), and no
`num_warps`, so 4 warps. The plan passes the Python ints 12, 23, 34, 1, 2, 3 for
`brand12_id`, `brand23_id`, `brand34_id`, `sm_grp`, `md_grp`, `lg_grp`. Triton's
JIT would specialize the value 1 of `sm_grp` as a constant; `sm_grp` is not a
`tl.constexpr` argument, so `TYPES` states it as `i32` like the others.

## Lengths and ranges

- `l_quantity_ptr`, `l_extendedprice_ptr`, `l_discount_ptr` (`float64`),
  `l_partkey_ptr` (`int64`), `ship_ok_ptr` (`int8`, computed by cuDF from
  `l_shipmode` and `l_shipinstruct`): columns of `lineitem`, `n_elements` rows
  each; `n_elements = l_quantity.numel()`, and `_query_core` returns before the
  launch when it is 0, so its least value is 1.
- `brand_map_ptr`, `container_map_ptr`, `size_map_ptr`: the `int32` maps that
  `_zero3_i32_kernel` and `_build_part_maps_kernel` filled, `map_size =
  max(p_partkey) + 1` entries each, so `max_partkey + 1`. For an empty `part`,
  the maps have 1 entry and `max_partkey` is 0, which also gives `max_partkey + 1`.
- `max_partkey`: `int(df_part["p_partkey"].max())`, 2,000,000 at scale factor 10,
  least value 0.
- `l_partkey_ptr` is `read`: its values are foreign keys into `part`, and the plan
  does not bound them; the kernel checks `0 <= partkey <= max_partkey` before the
  map loads.
- `brand12_id`, `brand23_id`, `brand34_id`, `sm_grp`, `md_grp`, `lg_grp`: the
  constants 12, 23, 34, 1, 2, 3.
- `out_ptr`: `torch.zeros(1, dtype=torch.float64)`, one element, which the kernel
  adds to with `tl.atomic_add` (`readwrite`).
