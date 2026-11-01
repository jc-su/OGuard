# Classification of the refused corpus kernels

The initial checker (scratch build), constant sizes, bindings from `run_corpus.py`. Source of truth for the rows: `classes_data.py` (it also computes the counts below).

This file records the classification at the start of the project, the Opus and Triton kernels that joined later, and the held-out tasks. The gate's current decisions are in `decisions.tsv`, which `tests/gate/test_corpus.py` asserts: on the development tasks the gate accepts 72 of the 76 kernels that are safe under their contract and none of the 56 with a defect, and on the held-out tasks 31 of the 66 safe kernels and none of the 46 with a defect.

## Scope

- 72 kernels: 11 accepted, 55 refused, 6 not bound. `run_corpus.py` cannot bind `segmented_sum` (the length of `values` is `offsets[g]`). `run_segsum.py` binds it with `values` of n = 1000 rows and otherwise the same rules; all 6 are refused (`results_segsum.jsonl`). This file has 61 rows: the 55 refusals and the 6 `segmented_sum` kernels.
- Evidence per kernel: the source, the PTX, and the checker's failing accesses. `diag/<model>_<task>_<toolchain>.diag` is the diagnosis at the final variant. For nvcc, `diag/*.pinu.diag` (pin_block and unsigned loads, no operand swap) and `diag/*.u.diag` (unsigned loads only) remove two harness artifacts (see "Harness findings"). `summ.py` prints the failing accesses with their address forms. `atoms.py` prints the facts of the registers in an address. `idioms.py` lists the PTX idioms per kernel.

## Verdicts

The verdict is against the task contract in `TASKS.md`, with the facts that the gate can establish. The catalog establishes sizes, scalar ranges, and the value range of an index input: `sel` in [0, n - 1], string offsets in [0, C], scatter offsets in [0, total], and segment offsets in [0, N]. It does not establish the contents of an output or inout on entry (DESIGN 5.4), or relations between buffers (for example, offsets that increase, or offsets that are the prefix sum of flags). The block size is 256 (the binding). The grid is any size up to the hardware limit.

- `C`: a defect that is reachable with inputs that meet the whole task contract and a launch that the signature allows.
- `E`: a defect that is reachable only when a task precondition that the gate cannot establish is violated. The gate must refuse these kernels (DESIGN 5.4 argues this for the counter case). Tenant data controls the precondition.
- `limit`: the kernel is safe under every fact that the gate can establish, and the checker cannot prove it.

## Feature codes

For a limit, the column lists the features that the kernel needs. For a defect, the "after the fix" column lists the features that the fixed kernel needs.

| Code | Feature |
| --- | --- |
| AT | atomics (read-write role) |
| RB | read-back of an output or inout with plain loads (read-write role; the same mechanism as AT in DESIGN 5.7) |
| MK | register mask `x & (capacity - 1)`, also through the join of the probe loop (R4 + R6). Requires `capacity >= 1` in the binding: with capacity 0 the mask is -1 |
| GS | grid-stride loop with a register step `ntid * nctaid` whose body the compiler does not unroll: the loop still tests `i < n` on every iteration (R5) |
| GS* | grid-stride loop with a simple body, compiled as a counted loop. Both NVVM and nvcc compute the trip count with a division (`div.u64`, or `div.u32` when both operands fit): nvcc divides `n - i - 1` by the stride s, and NVVM divides `max(i + s, n) + s - (i + s) - 1`. They run the remainder iterations with a down-counter and unroll the rest 4 times, testing `i < n` once per 4 iterations. The loads at `i + s`, `i + 2s`, and `i + 3s` have no guard, so R5 alone does not prove them |
| LG | launch geometry beyond a constant `%ntid`: here `%tid.x < 256`, from the fixed block size (`counts[tid]`) |
| SZ | a bound through a loaded offset or a derived size: `j < end <= C` (R1, transitive step), `i < shape - 1`. Plain `i < n` guards are not listed, because every kernel needs them once sizes are symbols |
| LR | a relation between two loop registers. NVVM and nvcc turn `for j in range(start, end)`, `for (p = begin; p < end; ++p)`, and the 32-step warp loops into counted loops, 4 times unrolled. A 64-bit pointer goes up while a down-counter (from `end - start`, or a trip count from a shift) goes to zero. The exit tests the counter, or tests the bound only once per 4 iterations. Numba's `range` loops also count down. The address bound is `ptr + counter = const`, which needs a zone or octagon fact. DESIGN 5.3 postpones this ("only if the corpus needs it"). The corpus needs it |
| DL | length-check relation: `start + j < end` follows from `end - start >= k` and `j < k` (starts_with) |
| SI | Numba's selp-by-predicate wrap term `i + (i < -1 ? shape : 0)`, which `+no_wraparound` does not remove, or Numba's floor-division and modulo idiom (`grid(1) // 32`, `tid % 32`) |
| SN | signed narrow load: `ld.s32` of an index |
| ZX | a guard on the zero extension of a 32-bit index. Once `%ntid` is 256, `cvt.u64.u32(mad.lo.s32(ctaid, 256, tid))` is a wrapping 32-bit form, and the initial checker no longer bounds it with the guard on the 64-bit value (R2 must carry the guard through `cvt`) |
| (SV) | stored-value range for an output that the signature marks `indexes_into`. The initial checker did not model stored values; OpGate's gate checks them since milestone 5 (DESIGN 5.4). After the fixes, every stored value is a thread index below a size, or `table_rows[slot]` (R3) |

Every row also needs the baseline B: `%ntid` as a constant, plus for Numba the `mul.wide` operand order and the and-form wrap term. These are the harness edits `pin_block`, `+wide_order`, and `+no_wraparound`, and they are not listed per row. No refused kernel needs shared-memory staging of index values, warp shuffles or ballots in an address, or an inlined device call. NVVM and nvcc inline every device function (no `call` in any PTX). Shuffles and ballots feed addresses only in fable's filter_select and hash_probe (the warp-aggregated output position), which are defects: after the two-sided guard, no fact about the shuffled value is needed. Fable's groupby reads keys back from shared memory, but they reach a global address only through `& mask`.

## Table

| Model | Task | Toolchain | Verdict | Trigger (defect) or features (limit) | Features after the fix | Fix |
| --- | --- | --- | --- | --- | --- | --- |
| haiku | filter_select | numba | E | count[0] = -(n + 1) on entry (the task does not say count is zero on entry): idx passes the one-sided `idx < n`, Numba adds n to the negative index, store to sel[-1] | AT (SV) | `if 0 <= idx < n` |
| haiku | filter_select | nvcc | E | count[0] = -1 on entry: int32 idx is sign-extended, passes `idx < n`, store to sel[-1] | AT ZX (SV) | `if (idx >= 0 && idx < n)` or an unsigned compare |
| haiku | groupby_sum | numba | limit | AT RB MK | - | none; all three are inherent to hash aggregation |
| haiku | groupby_sum | nvcc | C | no row count, guard is `i < gridDim.x*blockDim.x`: n = 1000 with 4 blocks of 256 reads group[1000..1023] and value[1000..1023] | AT RB MK ZX | add `n`, guard `i < n` |
| haiku | hash_build | numba | limit | RB MK | - | none (no atomics: racy but memory-safe) |
| haiku | hash_build | nvcc | C | no row count, guard is `i < gridDim.x*blockDim.x`: n = 1000 with 4 blocks reads keys[1000..1023] | RB MK ZX | add `n`, guard `i < n` |
| haiku | hash_probe | numba | C | store guard is `idx < capacity`, not out_cap (out_cap is not a parameter): capacity = 2048, out_cap = 100, 1000 matching probe keys write out_build/out_probe[100..999] | AT MK (SV) | pass out_cap, guard `0 <= idx < out_cap` |
| haiku | hash_probe | nvcc | C | same as Numba (`idx < capacity`); also out_count = -1 on entry stores to out_build[-1] | AT MK ZX | pass out_cap, guard `(unsigned)idx < out_cap` |
| haiku | histogram | numba | limit | AT | - | none |
| haiku | histogram | nvcc | C | no row count, guard is `i < gridDim.x*blockDim.x`: n = 1000 with 4 blocks reads keys[1000..1023] | AT ZX | add `n`, guard `i < n` |
| haiku | q6_sum | numba | limit | AT | - | none |
| haiku | q6_sum | nvcc | C | `n = gridDim.x*blockDim.x` replaces the row count: 1000 rows with 4 blocks read shipdate/discount/quantity/price[1000..1023] | AT GS ZX | add `n`, loop `i < n` |
| haiku | scatter_by_offsets | numba | E | flags[i] = 1 with offsets[i] = total (a value a prefix sum has only at unflagged rows); haiku's signature gives offsets no range, so any value | SN | guard `0 <= o < total`; declare offsets as an index input |
| haiku | scatter_by_offsets | nvcc | E | as Numba: flags[i] = 1 with offsets[i] = total writes out[total] | ZX | guard `(unsigned)o < total` |
| haiku | starts_with | numba | limit | SN SI SZ DL LR | - | none small; the signature must declare offsets as an index input |
| haiku | starts_with | nvcc | C | `n = gridDim.x*blockDim.x - 1`: 1000 strings with 4 blocks read offsets[1001..1023] and write match[1000..1022] | DL ZX | pass `n`, guard `i < n` |
| haiku | string_length | numba | C | TYPES declares lengths int64[:], the signature and task int32: 8-byte stores at 8*i pass the 4n-byte buffer for i >= n/2 (n = 1 is enough) | SN SI SZ | declare lengths `int32[:]` |
| haiku | string_length | nvcc | C | `n = gridDim.x*blockDim.x - 1`: 1000 strings with 4 blocks read offsets past n+1; int32 `pos += 4` also wraps for a string ending at >= 2^31 - 3 | SZ ZX | pass `n`; 64-bit pos |
| haiku | segmented_sum | numba | limit | SI SZ LR | - | none small |
| haiku | segmented_sum | nvcc | limit | SZ LR ZX | - | none small |
| sonnet | filter_select | numba | E | count[0] = n on entry: no guard on pos, store to sel[n] | AT (SV) | `if 0 <= pos < n` |
| sonnet | filter_select | nvcc | E | count[0] = n on entry: store to sel[n] | AT (SV) | `if ((unsigned)pos < n)` |
| sonnet | groupby_sum | numba | limit | AT MK | - | none |
| sonnet | groupby_sum | nvcc | limit | AT MK | - | none |
| sonnet | hash_build | numba | limit | AT MK (SV) | - | none |
| sonnet | hash_build | nvcc | limit | AT MK (SV) | - | none |
| sonnet | hash_probe | numba | E | out_count[0] = -(out_cap + 1) on entry (the task says zero on entry; the gate cannot establish it): pos passes `pos < out_cap`, Numba adds out_cap, store to out_build[-1] | AT MK (SV) | `if 0 <= pos < out_cap` |
| sonnet | hash_probe | nvcc | E | out_count[0] = -1 on entry: store to out_build[-1] | AT MK (SV) | `pos >= 0 &&` or an unsigned counter |
| sonnet | histogram | numba | limit | AT GS LG | - | none |
| sonnet | histogram | nvcc | limit | AT GS LG | - | none |
| sonnet | q6_sum | numba | limit | AT | - | none |
| sonnet | q6_sum | nvcc | limit | AT | - | none |
| sonnet | scatter_by_offsets | numba | E | flags[i] = 1 with offsets[i] = total (a prefix sum has that value only at unflagged rows): store to out[total]; the offsets range is [0, total] (DESIGN 10.3), so the store needs a guard | SN | guard `0 <= o < total` |
| sonnet | scatter_by_offsets | nvcc | E | as Numba: store to out[total] | SN | guard `(unsigned)o < total` |
| sonnet | starts_with | numba | limit | SN SI DL LR | - | none small |
| sonnet | string_length | numba | limit | SI SZ LR | - | none small |
| sonnet | string_length | nvcc | limit | SZ LR | - | none small (maybe `#pragma unroll 1`, not tested) |
| sonnet | segmented_sum | numba | limit | SN SI SZ LR | - | none small |
| sonnet | segmented_sum | nvcc | C | int32 `idx += 32` wraps for a segment ending at >= 2^31 - 31: g = 1, offsets = [0, 2^31 - 1] (16 GiB of values); the unrolled loop tests `idx < end` after the wrap and reads past the end | SZ LR | `long long idx` |
| fable | filter_select | numba | E | count[0] = n on entry: store to sel[n + rank], no guard | AT GS (SV) | guard `0 <= pos < n` |
| fable | filter_select | nvcc | E | count[0] = n on entry: store to sel[n + rank] | AT GS (SV) | guard `0 <= pos < n` |
| fable | gather | numba | limit | SN GS GS* | - | one row per thread with `if i < m` (the accepted form) |
| fable | gather | nvcc | limit | GS GS* | - | one row per thread with `if (i < m)` |
| fable | groupby_sum | numba | limit | AT RB MK GS | - | none |
| fable | groupby_sum | nvcc | limit | AT RB MK GS | - | none |
| fable | hash_build | numba | limit | AT RB MK GS (SV) | - | none (the read before the CAS needs RB) |
| fable | hash_build | nvcc | limit | AT RB MK GS (SV) | - | none |
| fable | hash_probe | numba | E | out_count[0] = -(out_cap + 1) on entry: pos = base + rank passes `pos < out_cap`, Numba adds out_cap, store to out_build[-1] | AT MK GS (SV) | guard `0 <= pos < out_cap` |
| fable | hash_probe | nvcc | E | out_count[0] = -1 on entry: pos = -1 + rank, store to out_build[-1] | AT MK GS (SV) | `pos >= 0 &&` |
| fable | histogram | numba | limit | AT GS | - | none |
| fable | histogram | nvcc | limit | AT GS | - | none |
| fable | projection | numba | limit | GS GS* | - | one row per thread with `if i < n` |
| fable | projection | nvcc | limit | GS GS* | - | one row per thread with `if (i < n)` |
| fable | q6_sum | numba | limit | AT SI GS GS* | - | one row per thread and a block reduction (sonnet's form) leaves AT |
| fable | q6_sum | nvcc | limit | AT GS GS* | - | one row per thread and a block reduction leaves AT |
| fable | scatter_by_offsets | numba | E | flags[i] = 1 with offsets[i] = total (inside the honest range [0, total]) writes out[total] | SN GS GS* | guard `offsets[i] < total` |
| fable | scatter_by_offsets | nvcc | E | flags[i] = 1 with offsets[i] = total writes out[total] | GS GS* | guard `offsets[i] < total` |
| fable | starts_with | numba | E | length is uint32 `offsets[i+1] - begin`, which wraps when offsets decrease: n = 2, C = 10, offsets = [10, 0, 10], k = 5 keeps ok true and reads chars[10..14] | SI GS GS* DL | compute the length in int64 |
| fable | starts_with | nvcc | limit | GS GS* DL | - | none small |
| fable | string_length | numba | limit | SN SI GS SZ LR | - | none small |
| fable | string_length | nvcc | C | warp path, int32 `p = p0 + lane` wraps: n = 1, offsets = [1, 2^31 - 1] (2 GiB of chars, average >= 64): lane 31 of the last chunk gets p = -2^31, passes `p < end`, and the unrolled +96 load reads chars[2^31] | GS SZ LR | uint32 or 64-bit p0 and p (as fable's Numba version) |
| fable | segmented_sum | numba | limit | SN SI GS SZ LR | - | none small |
| fable | segmented_sum | nvcc | C | warp path, int32 `p += 32` wraps: g = 1, offsets = [0, 2^31 - 1] (16 GiB of values, average >= 32); the unrolled loop tests `p < end` after the wrap and reads past the end | GS SZ LR | 64-bit p |

## Counts

Verdicts, all 63 rows (61 of the constant-size run, and sonnet's two scatter_by_offsets kernels, refused at symbolic sizes): 34 limit, 17 E, 12 C.

| Model | C | E | limit | Rows |
| --- | --- | --- | --- | --- |
| haiku | 9 | 4 | 7 | 20 |
| sonnet | 1 | 6 | 12 | 19 |
| fable | 2 | 7 | 15 | 24 |

Without `segmented_sum` (the 55 refusals of `results_*.jsonl`): 30 limit, 15 E, 10 C.

Classes of defect:

- No row count (C, 6, all haiku nvcc): the kernel derives n from the grid (`gridDim.x*blockDim.x`) in q6_sum, hash_build, groupby_sum, histogram, string_length, and starts_with. Haiku's signatures for these tasks also have no size parameter.
- Wrong bound or type (C, 3, haiku): hash_probe guards the store with `capacity`, not `out_cap` (Numba and nvcc); string_length stores int64 into an int32 column (Numba).
- Signed 32-bit wrap on large inputs (C, 3): sonnet and fable nvcc segmented_sum (a segment ending at 2^31 - 31 or later; 16 GiB of values) and fable nvcc string_length (a string ending at 2^31 - 30 or later; 2 GiB of chars). The Numba versions use int64 or uint32 and are safe.
- Counter on entry (E, 10): every filter_select kernel (6) and the hash_probe kernels of sonnet and fable (4). Four have no guard. Six have the one-sided guard of DESIGN 5.4: haiku filter_select (both), and sonnet and fable hash_probe (both). None has a two-sided guard or an unsigned compare.
- Offsets relation (E, 7): scatter_by_offsets trusts that `flags[i] != 0` implies `offsets[i] < total` (haiku 2, fable 2, sonnet 2; sonnet's were accepted at constant sizes, where the adapter gave the offsets the range of an index input, and are refused at symbolic sizes, where an offsets buffer's range is `[0, total]`). Fable's Numba starts_with trusts that offsets increase.

Features. The limit column counts only the 34 limit rows. The after-fix column counts the 27 defect rows after the fix. B is in all 61 rows.

| Feature | Limit | Defect after fix | All |
| --- | --- | --- | --- |
| AT atomics | 19 | 15 | 34 |
| GS grid-stride, register step | 17 | 10 | 27 |
| MK register mask | 10 | 8 | 18 |
| LR loop-register relation | 9 | 3 | 12 |
| SZ bound through a loaded offset or derived size | 8 | 5 | 13 |
| SI selp wrap / floor division | 8 | 2 | 10 |
| GS* counted, unrolled grid-stride | 7 | 3 | 10 |
| RB read-back of an output | 6 | 2 | 8 |
| SN signed narrow load | 6 | 3 | 9 |
| DL length-check relation | 3 | 2 | 5 |
| LG `%tid < ntid` | 2 | 0 | 2 |
| ZX guard through `cvt.u64.u32` | 1 | 9 | 10 |
| (SV) stored-value range, optional | 4 | 11 | 15 |
| shared staging, warp shuffles in addresses, device call | 0 | 0 | 0 |

## Features that unlock the most kernels

The analysis counts a kernel as unlocked when its whole feature set is implemented. AT and RB are one mechanism (the read-write role, RW). The counts come from `classes_data.py` (exhaustive over feature subsets).

Of the 34 limits, with nothing beyond B:

| Features | Limits unlocked |
| --- | --- |
| RW | 4 |
| RW + MK | 10 |
| RW + MK + GS | 16 |
| RW + MK + GS + GS* | 20 |

If the idiom rules SI, SN, and ZX are also in place (planned in DESIGN 5.5 and R2; cheap), RW + MK + GS + GS* unlocks 22 of 34. Adding LR + SZ (the string and segment loops) makes 29. LG (sonnet histogram, 2) and DL (starts_with, 3) complete the 34.

Of all 61 rows, with each defect fixed and SI, SN, and ZX in place: RW + MK + GS + GS* gives 42, and with LR + SZ, 54.

The four features that unlock the most kernels:

1. The read-write role (atomics and read-back): needed by 20 of the 34 limits (AT 19, RB 6, overlapping), and by 16 fixed defects.
2. The register mask (R4 through the probe-loop join): every hash kernel (10 limits, 8 fixed defects).
3. Grid-stride loops with a register step (R5): 17 limits. Every fable kernel uses them.
4. The compiled shape of the simple grid-stride loop (GS*): 7 limits. This shape needs a fact about a `div.u64` trip count: `i + (t - 1) * s <= n - 1` when `t = floor((n - 1 - i) / s) + 1`. That is outside R1-R6. A source workaround exists: one row per thread, as the accepted gather and projection kernels use. So the design can choose between the rule and the repair message.

The next item is the loop-register relation LR with SZ: 9 limits, which are all the limit kernels of string_length and segmented_sum plus the Numba starts_with of haiku and sonnet, on both toolchains. No reliable source workaround exists for Numba, because Numba's `range` loops count down and NVVM unrolls them. This is the sparse-zone case that DESIGN 5.3 defers; the corpus needs it.

## Accepted kernels

The 11 accepted kernels: haiku gather and projection (Numba, nvcc); sonnet gather, projection, and scatter_by_offsets (Numba, nvcc); and sonnet starts_with (nvcc). The Numba ones were accepted after `+no_wraparound` and `+unsigned_loads`. These edits are exact when the index is at least 0, and the index ranges guarantee that.

None is unsafe under its binding. Two are unsafe under the honest binding, and one binding rule of the harness is wrong:

- Sonnet scatter_by_offsets (Numba, nvcc) is accepted only because `indexes_into: out` gave offsets the range [0, total - 1]. An exclusive prefix sum has the value total at every row after the last flagged row, so real inputs fail that range. The catalog's check would refuse the launch, so no memory access happens. With the honest range [0, total], both kernels are E (flags[i] = 1 and offsets[i] = total write out[total]), and the gate must refuse them. This is a fault in how the binding is derived, not a soundness fault of the checker.
- The same rule gives string offsets [0, C - 1], but every valid column has offsets[n] = C. A kernel that reads `chars[offsets[i+1]]` (one past its string) would be accepted under that range. Sonnet starts_with nvcc, the only accepted string kernel, is also safe under [0, C]. It reads `chars[start + j]` with `start + j < end <= C`; `end - start` in int32 cannot overflow for values in [0, 2^31 - 1]; and decreasing offsets give a negative length and no reads. For offsets arrays, the rule must give [0, rows(target)].

## Harness findings that change the diagnoses

1. `classify.py` misses failures that wrap across lines. Its regex expects `, Some` after one space, so it drops most atomics (`KRmw`) and gives empty lists (haiku histogram and q6 Numba, sonnet groupby and q6 Numba). `summ.py` parses the same output with the whitespace normalized.
2. `wide_const_second` swaps the operands of every `mul.wide` with three registers. For nvcc it moves the pinned 256 to the first operand. After the swap, sonnet gather, projection, scatter_by_offsets, and starts_with nvcc (accepted at `pin_block`) are refused (`check_swap.py`). The nvcc diagnoses here use `pin_block` and unsigned loads without the swap.
3. `pin_block` breaks haiku's nvcc index `mad.lo.s32 %r, %ctaid, 256, %tid; cvt.u64.u32`. Haiku gather and projection nvcc are accepted as compiled and refused with `%ntid` = 256 (`check_pin.py`). This is feature ZX. A constant `%ntid` in the design needs R2 across `cvt.u64.u32`, or it removes coverage.
4. `unsigned_narrow_loads` does not match `ld.global.s8 %rs` (16-bit destination). It affects only a value in haiku string_length nvcc.
5. No refused nvcc kernel is accepted with `pin_block` and unsigned loads, or with unsigned loads alone (`diag/nvcc2_log.txt`).

## Opus (recorded 2026-09-27)

The 24 Opus kernels are classified against OpGate's gate at NVVM level 1 (`numba.O1.ptx`, `kernel.O1.ptx`), with the `opus` rows of `roles.tsv` and the facts the gate establishes: sizes, scalar ranges, index-input ranges, block 256, and any grid up to the hardware limit. The gate accepts 7: nvcc `gather`, `histogram`, `projection`, `starts_with`, and `string_length`, and both `hash_build`. Of the 17 refusals, 1 is `C`, 6 are `E`, and 10 are limits. Each limit was checked by a variant with one change, compiled by the corpus's front ends and decided by the gate; the fix column states the change that the gate accepts.

| Model | Task | Toolchain | Verdict | Trigger (defect) or features (limit) | Fix |
| --- | --- | --- | --- | --- | --- |
| opus | filter_select | numba | E | count[0] = 1 on entry, n = 1, quantity = [0.0], threshold = 1.0: the tile's atomic add returns 1, so the store goes to sel[1]; with count[0] = -(n + 1), Numba adds n and stores to sel[-1] | store only `if 0 <= pos < n` |
| opus | filter_select | nvcc | E | count[0] = 1 on entry, as for Numba: store to sel[1]; with count[0] = -1, to sel[-1] | store only `if (pos >= 0 && pos < n)` |
| opus | gather | numba | limit | RG: `for i in range(cuda.grid(1), m, cuda.gridsize(1))` | a `while i < m` loop is accepted |
| opus | groupby_sum | numba | limit | WA: `warp_start = cuda.grid(1) - lane`, then group[j] and value[j] with `j = warp_start + lane` | `warp_start = cuda.blockIdx.x * cuda.blockDim.x + (cuda.threadIdx.x & 0xE0)` is accepted |
| opus | groupby_sum | nvcc | limit | WA, as for Numba | `warp_start = grid(1) & ~31LL` is accepted |
| opus | hash_probe | numba | E | out_count[0] = -(out_cap + 1) on entry, out_cap = 1, m = 1, a probe key in the table: position = -2 passes `position < out_cap`, and Numba's wrap-around stores to out_build[-1] and out_probe[-1] | `if 0 <= position < out_cap:` |
| opus | hash_probe | nvcc | E | out_count[0] = -1 on entry: stores to out_build[-1] and out_probe[-1] | `if (position >= 0 && position < out_cap)` |
| opus | histogram | numba | limit | RG twice: `range(tid, BINS, cuda.blockDim.x)` and `range(cuda.grid(1), n, cuda.gridsize(1))` | both loops as `while` loops are accepted |
| opus | projection | numba | limit | RG | the `while` loop is accepted |
| opus | q6_sum | numba | limit | RG | the `while` loop is accepted |
| opus | q6_sum | nvcc | limit | BF: `lane < WARPS ? warp_sums[lane] : 0.0` with `lane = threadIdx.x % 32`; the guard `(tid & 31) <= 7` does not narrow the bit field that addresses the 64-byte `warp_sums` | the guard on `threadIdx.x` is accepted |
| opus | scatter_by_offsets | numba | E | flags = [1], offsets = [1], total = 1: store to out[1]; the value total lies in the range [0, total] of an offsets buffer, and a prefix sum of 0 or 1 flags has it only at unflagged rows | store only `if flags[i] != 0 and offsets[i] < total` (accepted) |
| opus | scatter_by_offsets | nvcc | E | as for Numba | `if (flags[i] != 0 && offsets[i] < total)` (accepted) |
| opus | segmented_sum | numba | limit | RG, SI, LR, SH, BI, ST, SZ | none small |
| opus | segmented_sum | nvcc | C | warp mode: `int64_t i = offsets[j] + lane` adds two `int` values, which wrap before the conversion; g = 2, offsets = [2^31 - 65, 2^31 - 1, 2^31 - 1] (16 GiB of values), grid 1: lane 1 of warp 1 reads values[-2^31] | `static_cast<int64_t>(offsets[j]) + lane` |
| opus | starts_with | numba | limit | RG | the `while` loop is accepted |
| opus | string_length | numba | limit | RG, and NR: the inner `range(offsets[i], offsets[i + 1])` loop inside the grid-stride loop | both loops as `while` loops are accepted |

New feature codes:

| Code | Construct |
| --- | --- |
| RG | Numba `range(start, stop, step)` with a register step. Numba's own lowering computes the trip count with `div.s64`, adds 1 for a remainder, counts it down, and never compares the index with stop, also at NVVM level 1. The bound `start + (t - 1) * step <= stop - 1` needs the product of two registers. Only Opus writes this loop; the other models write `while` loops |
| WA | a warp-aligned start `grid(1) - lane`: the checker bounds `tid - (tid & 31)` below by -31 |
| BF | a guard against a constant on a bit field `(tid & 31) <= 7` that does not narrow the bit field |
| NR | a Numba `range(start, end)` loop over a string nested in another loop |
| SH | a value of another lane from `shfl.sync` that bounds a loop or is a term of an address |
| BI | a `ballot` of other lanes' predicates, whose population count is a term of an address |

The flags of `scatter_by_offsets` are read as 0 or 1, as for the first corpus. The task does not restrict the int8 flags; with a negative flag, a prefix sum of flags can reach total at a flagged row, and each scatter kernel without a `< total` guard is then `C`.

Since this record, the bit rules of design decision 24 accept the WA and BF limits (`groupby_sum` of both toolchains and nvcc `q6_sum`), and the front end's lowering of `range` (decision 25) accepts the six RG kernels unchanged. Of the 10 limits, only Numba `segmented_sum` stays refused, by design: its warp mode bounds a long segment by `shfl.sync` values and indexes its output by a `ballot` bit, both values of another lane, which the gate reads as arbitrary values, as for Triton's RD.

## Triton (recorded 2026-09-27)

The 36 Triton kernels of Fable, Opus, and Sonnet ([TRITON.md](TRITON.md); Haiku's break the interface and are not in the table) are classified against OpGate's gate with `triton.ptx`, the dynamic shared memory of `triton.json`, and the rows of each model and task in `roles.tsv`. The facts are those the gate establishes: sizes, scalar ranges, index-input ranges, a block of `32 * NUM_WARPS` threads, and any grid up to 2^31 - 1 programs in x. Before the Triton rules of design decision 24 (SB, PL, and WS below, and `xor.pred`), the gate accepted 7: Fable `groupby_sum`, `histogram`, `projection`, and `q6_sum`, and Opus `histogram`, `projection`, and `q6_sum`. Of the 29 refusals, 14 are `C`, 6 are `E`, and 9 are limits. Each fix was compiled from an edited copy by the corpus's Triton front end and decided by the gate before those rules; where the gate still refused it, the column names the features it still needed. The decision table has the gate's current decisions.

| Model | Task | Toolchain | Verdict | Trigger (defect) or features (limit) | Fix |
| --- | --- | --- | --- | --- | --- |
| fable | filter_select | triton | E | count[0] = 1 on entry, n = 1, quantity = [0.0], threshold = 1.0: the tile's `atomic_add(count, num)` returns 1, so `pos = base + cumsum - 1` is 1 and the store goes to sel[1]; with count[0] = -1, to sel[-1] | store only where `(pos >= 0) & (pos < n)`; still refused: PG, SB |
| fable | gather | triton | limit | PL: `src = tl.load(sel + rows, mask=in_bounds, other=0)` has no range, so `col + src` is not bounded | none in the source |
| fable | hash_build | triton | limit | SB: the store of `rows.to(tl.int32)` to `table_rows`; every address is proved | none in the source |
| fable | hash_probe | triton | E | out_count[0] = -1 on entry, m = 1, out_cap = 1, a probe key in the table: `pos = -1` passes `pos < out_cap`, stores to out_build[-1] and out_probe[-1] | `writable = found & (pos >= 0) & (pos < out_cap)`; still refused: PG, SB |
| fable | scatter_by_offsets | triton | E | flags = [1], offsets = [1], total = 1: store to out[1] | store only where `dst < total`; with `dst >= 0` also, still refused: SB |
| fable | segmented_sum | triton | C | `for j0 in range(0, longest, V)` counts in int32, since `longest = tl.max(end - begin)` is int32: g = 1, offsets = [0, 2^31 - 1], one program; `j0` wraps to -2^31 and lane 0 reads values[-2^31] | `lens = end.to(tl.int64) - begin`; still refused: PL, RD |
| fable | starts_with | triton | limit | PL, FL, DL | none small |
| fable | string_length | triton | C | the same int32 `j0` wrap: n = 1, offsets = [0, 2^31 - 1], one program; lane 0 reads chars[-2^31] | `lens = end.to(tl.int64) - begin`; still refused: PL, RD |
| opus | filter_select | triton | E | count[0] = 1 on entry: `slot = 1`, `rank = 0`, store to sel[1]; with count[0] = -1, to sel[-1] | store only where `(slot + rank >= 0) & (slot + rank < n)` with a 64-bit `slot`: the address is proved; still refused: SB |
| opus | gather | triton | limit | PL, as Fable's | none in the source |
| opus | groupby_sum | triton | limit | SW: a `st.shared.v4.b32` of `tl.sort` whose address has an `xor` term; WS | without the select and the sorted runs, accepted |
| opus | hash_build | triton | limit | WS: `slot = tl.where(pending, (slot + 1) & mask, slot)`; SB | `slot = (slot + 1) & mask` removes WS; SB remains |
| opus | hash_probe | triton | E | out_count[0] = -1 on entry: `position = -1` passes `position < out_cap`, stores to out_build[-1] and out_probe[-1]; the first refused access is the WS limit | `write = found & (position >= 0) & (position < out_cap)`; still refused: WS, PG, SB |
| opus | scatter_by_offsets | triton | E | flags = [1], offsets = [1], total = 1: store to out[1] | store only where `position < total`; with `position >= 0` also, still refused: SB |
| opus | segmented_sum | triton | limit | GA: the second row of a lane, `base + lane + 32`, is guarded as `base \| (lane \| 32)`; PL; RD | none small |
| opus | starts_with | triton | limit | PL, FL, DL | none small |
| opus | string_length | triton | limit | GA, PL, RD, as `segmented_sum` | none small |
| sonnet | filter_select | triton | C | 32-bit offsets, BLOCK = 1024, a grid of 2^21 + 1 programs: program 2^21 reads quantity[-2^31]; also E: no guard on `pos` | a 64-bit `pid` and store only where `(pos >= 0) & (pos < n)`; still refused: SB |
| sonnet | gather | triton | C | 32-bit offsets, grid 2^21 + 1: reads sel[-2^31] | a 64-bit `pid`; still refused: PL |
| sonnet | groupby_sum | triton | C | 32-bit rows, ROWS = 128, grid 2^24 + 1: reads group[-2^31] | a 64-bit `pid`; still refused: WS, MT; with a 64-bit mask and no select, accepted |
| sonnet | hash_build | triton | C | 32-bit rows, grid 2^24 + 1: reads keys[-2^31] | a 64-bit `pid`; still refused: WS, SB |
| sonnet | hash_probe | triton | C | 32-bit rows, grid 2^24 + 1: reads probe_keys[-2^31]; also E: `pos = -1` passes `pos < out_cap` | a 64-bit `pid` and `(pos >= 0) & (pos < out_cap)`; still refused: WS, SB |
| sonnet | histogram | triton | C | 32-bit offsets, BLOCK = 4096, grid 2^19 + 1: reads keys[-2^31] | a 64-bit `pid`; still refused: PK; with BLOCK = 1024 also, accepted |
| sonnet | projection | triton | C | 32-bit offsets, grid 2^21 + 1: reads price[-2^31], writes out[-2^31] | a 64-bit `pid` (accepted) |
| sonnet | q6_sum | triton | C | 32-bit offsets, grid 2^21 + 1: reads shipdate[-2^31] | a 64-bit `pid` (accepted) |
| sonnet | scatter_by_offsets | triton | C | 32-bit offsets, grid 2^21 + 1: reads flags[-2^31]; also E: no `< total` guard | a 64-bit `pid` and `pos < total`; with `pos >= 0` also, still refused: SB |
| sonnet | segmented_sum | triton | C | no `pid < g` guard: g = 1, offsets = [0, 0], a grid of 2: program 1 reads offsets[2] and writes sums[1]; also an int32 `pos` that wraps near 2^31 | `if pid < g:` and 64-bit `start`, `end` (accepted; either alone is refused) |
| sonnet | starts_with | triton | C | 32-bit rows, grid 2^24 + 1: reads offsets[-2^31] | a 64-bit `pid`; still refused: PL, FL, DL |
| sonnet | string_length | triton | C | 32-bit rows, grid 2^24 + 1: reads offsets[-2^31] | a 64-bit `pid`; still refused: PL, FL |

| Model | C | E | limit | Refused |
| --- | --- | --- | --- | --- |
| fable | 2 | 3 | 3 | 8 |
| opus | 0 | 3 | 6 | 9 |
| sonnet | 12 | 0 | 0 | 12 |
| all | 14 | 6 | 9 | 29 |

Classes of defect: the 32-bit program offset (C, 11, all of Sonnet's kernels but `segmented_sum`; only a planner's grid larger than `cdiv(n, BLOCK)` reaches it, and no Fable or Opus kernel has it, since both write `tl.program_id(0).to(tl.int64) * BLOCK`); a missing program guard (C, Sonnet `segmented_sum`); an int32 loop counter (C, Fable `segmented_sum` and `string_length`, reached with the model's own grid); a counter on entry (E, 4); the offsets relation (E, 2).

New feature codes, with the count of limits and of defects after their fix that need each:

| Code | Construct | Limits | Defects after fix |
| --- | --- | --- | --- |
| PL | the value of a masked load, `tl.load(p, mask=m, other=v)`: `mov r, v; @m ld r, [a]`; rule R3 gave an index input's range only to an unguarded load | 6 | 8 |
| SB | Triton's store `st.global.b32 [a], { %r }`: the translation read the one-element vector as no value, so a store to an index output failed the stored-value check | 2 | 10 |
| WS | `tl.where(active, (slot + 1) & mask, slot)` on a hash slot: a `selp` of two registers, whose fact is the join of theirs and drops R4's `slot - capacity <= -1` | 2 | 4 |
| FL | a mask flag carried through a loop, `ok = valid & (end - begin >= k)`, whose join at the loop head forgets the comparison | 2 | 2 |
| RD | a reduction across lanes (`tl.max`, `tl.min`, `tl.sum`) that bounds a loop or is a term of an address | 2 | 2 |
| DL | the length check before a prefix loop through a masked load | 2 | 1 |
| PG | a guarded warp-aggregated position `base + rank`, `base` broadcast through shared memory with `ld.shared.s32` into a 64-bit register | 0 | 3 |
| GA | the alignment of a grid-stride loop register, `base` of `tl.range(pid * 64, g, nprog * 64)`, which the loop head drops, so `base \| (lane \| 32)` is not read as a sum | 2 | 0 |
| SW | Triton's swizzled shared-memory layout of `tl.sort`: an `xor` of thread-index bit fields in a shared address, which the instruction set does not have | 1 | 0 |
| MT | a hash mask at 32 bits, `(capacity - 1).to(tl.int32)`, which R4 does not relate to the 64-bit capacity | 0 | 1 |
| PK | predicates packed into an integer: with 32 elements per thread, LLVM packs the masks into a 16-bit register and unpacks each with `shr`, `and`, `setp.ne` | 0 | 1 |

PL and SB come from Triton's code shape, not from the models' choices: every masked `tl.load` compiles to `mov r, other; @mask ld r, [a]`, and every `tl.store` to `st.global.b32 [a], { r }`.

Since this record, the gate has PL, SB, WS, and SW (design decision 24) and PG and MT (decision 26); FL, RD, and PK stay refused by design, and GA is not needed, since both kernels that have it are also RD ([lowering catalog](../../../docs/LOWERING.md)). The fixed `filter_select` of Fable and Opus, Fable's fixed `hash_probe`, and Sonnet's fixed `groupby_sum` are accepted; Opus's fixed `hash_probe` stays refused, since its stored row index is guarded by a `found` flag that the probe loop sets only for rows below `m`, a flag that implies a guard across a join, which is refused by design.

## Held-out tasks (recorded 2026-09-28)

The 81 held-out kernels ([HELDOUT.md](HELDOUT.md)) that the gate refuses, of the 112 in the decision table, are classified against the task contracts with the classes and method above. The gate, the front ends, and the claim language are those that decided the development tasks, unchanged. The facts the checker may assume are those that the claim and the launch establish: each size symbol in `[minimum, 2^31 - 1]`, each scalar in its stated range, `codes` in `[0, d - 1]`, `run_ends` in `[0, n]`, `offsets` in `[0, chars]`, a block of 256 threads for Numba and CUDA C++ and `32 * NUM_WARPS` for Triton, the dynamic shared memory of the binding (none for Numba and CUDA C++, whose front ends launch with none), and every grid up to `GRID_LIMIT`. The bitmap `valid` has a size symbol of its own, with no stated relation to `n`. The contents of an output on entry, sortedness, and the relation `run_ends[r - 1] = n` are not facts.

Evidence: the source, the PTX the gate checked, and `eval.corpus.diagnose` for all but six large Triton kernels, whose diagnosis did not finish in 15 minutes; those rows rest on the source and the PTX. Every C and E row names a trigger that was run: each buffer its own allocation of its exact size, `compute-sanitizer --tool memcheck` (CUDA 12.9) on the RTX A6000, CUDA C++ through a `main` that includes the model's `kernel.cu`, Numba through `cuda.jit(kernel)` from its `numba.py`, and Triton through its `kernel[grid]`; each run gave the predicted out-of-bounds access. With no dynamic shared memory, memcheck flags a shared access only from offset `0x400`.

A counter or cursor defect reached by a grid with more than one block in y or z is C, since the planner is untrusted and the guarantee covers every grid; each of these kernels indexes by `x` only, so a second row of blocks repeats the work and overruns a count that is zero on entry. The development kernels' counter defects above are recorded as E, from a count out of range on entry; that classification did not consider grids with more than one block in y or z. A kernel that is safe under its own launch but uses dynamic shared memory in Numba or CUDA C++ is a limit (DS): the gap is the front end's statement, not the kernel.

| Model | Task | Toolchain | Class | Trigger (defect) or features (limit) | Ran |
| --- | --- | --- | --- | --- | --- |
| fable | merge_sorted | numba | E | `b` not sorted: m = p = 8192, `a` all 0, `b[0..2046] = 1`, `b[2047..8191] = -1`, the run's grid of 16 blocks. In block 3 the two merge-path searches return a0 = 3072 and a1 = 0, so na = -3072 and nb = 4096; thread 0 reads `s[na + j]` = `s[-3072]`, which Numba's wraparound turns into `s[-2048]`, 16 KiB below the shared array `s`. With sorted inputs 0 <= na <= 1024 and every access is in bounds. The refused `call` is `merge_path`, whose search is a BS limit | yes: invalid shared read at 0xffffc000, block 3 |
| fable | merge_sorted | nvcc | E | as Numba: `s[-3072]`, 24 KiB below `s` | yes: invalid shared read at 0xffffa000, block 3 |
| fable | merge_sorted | triton | limit | KS RD BS. For any contents of `a` and `b`: the k-ary search keeps `lo <= hi` inside `[max(d0 - p, 0), min(d0, m)]` (the count c is at most the number of lanes with `pos < hi`), so `i0` lies in `[lo, hi]`; the lane search keeps `mid` in `[lo_v, hi_v)` with `lo_v >= t - nb` and `hi_v <= min(t, na)`; the final loads are masked by `i < na` and `j < nb`, the store by `d0 + t < m + p` | no |
| fable | radix_scatter | numba | C | cursors meet the contract: n = 1, keys = [0], shift = 0, `cursors[0] = 0`, the other cursors 1; grid (1, 2, 1). The kernel indexes by `blockIdx.x` only, so both blocks rank the key and claim a slot of byte 0; the second stores `out[1]`. Also E with a 1-D grid: `cursors[0] = 1` on entry stores `out[1]` | yes (both triggers) |
| fable | radix_scatter | nvcc | C | as Numba | yes (both triggers) |
| fable | radix_scatter | triton | C | as Numba with grid (1, 2) programs: n = 2, keys = [0, 0], `cursors[0] = 0`, the others 2; the second program stores `out[2]` and `out[3]`. Also E with one program: `cursors[0] = 2` on entry stores `out[2]` and `out[3]` | yes (both triggers) |
| fable | rolling_sum | numba | limit | MM: `e = t0 + tile - h <= 255 + window <= 1279`, below the 1280 elements of `s`; `h + l < h + e = t0 + tile <= n`; `s[l - window]` is read only when `l >= window` | no |
| fable | rolling_sum | nvcc | limit | MM, as Numba; `e` is also truncated to 32 bits, which is exact for e in [1, 1279] | no |
| fable | run_end_decode | numba | E | the last run end is not n (the claim gives `run_ends` only the range [0, n]): r = 1, n = 2, `run_ends = [1]`, grid 1. `bounds[1] = upper_bound(row 1) + 1 = 2 > r`, so thread 1 searches [0, 2) and reads `run_ends[1]`, then `run_values[1]` or `[2]`. With `run_ends[r - 1] = n`, `bounds[1] <= r` and j <= r - 1 | yes: invalid read at `run_ends + 4` |
| fable | run_end_decode | nvcc | E | as Numba | yes: invalid read at `run_ends + 4` |
| fable | run_end_decode | triton | E | same input: the lifting takes cand = 1 because `run_ends[0] = 1 <= 1`, so j = r = 1 and row 1 reads `run_values[1]`. r = 0 with n >= 1 reads `run_values[0]` as well | yes: invalid read at `run_values + 8` |
| fable | sorted_contains | numba | limit | BS: `lo` and `hi` stay in [0, s] with `mid` in [lo, hi); `members[lo]` is read only when `lo < s` | no |
| fable | sorted_contains | nvcc | limit | BS | no |
| fable | sorted_contains | triton | limit | BL: `j` starts at 0 and only takes `cand = j + step` with step >= 1; the loop load is masked by `cand <= s` and the final one by `j < s`, so only `j >= 0` is needed, which the select of `j` and `j + step` loses (the diagnosis shows no fact for `j`) | no |
| fable | string_contains | numba | limit | LR SZ DL: `pos` runs from `s` to `e - k` with `e <= chars`, so `chars[pos + j]` with `j < k` lies below `e`; `pattern[j]` with `j < k` | no |
| fable | string_contains | nvcc | limit | LR SZ DL, as Numba | no |
| fable | string_contains | triton | limit | FL LR DL SZ: the loads are masked by `ok = (p + k <= length)`, with `length = end - beg`, and run over `q < k`, so `beg + p + q < end <= chars`; the checker loses the relation of the loop counter `q` to `k` and the comparison in `ok` inside the `q` loop | no |
| fable | valid_rows | numba | C | count zero on entry, n = 1, valid = [1], grid (1, 2, 1): both blocks take a slot and the second stores `rows[1]`. Also E with a 1-D grid: `count[0] = 1` on entry stores `rows[1]`. The refused load `valid[i >> 3]` is itself a QS limit | yes (both triggers) |
| fable | valid_rows | nvcc | C | as Numba; the refused load is the QS limit | yes (both triggers) |
| fable | valid_rows | triton | C | as Numba with grid (1, 2) programs: n = 2, valid = [3]; the second program stores `rows[2]` and `rows[3]`. Also E: `count[0] = 1` on entry. The refused load is the QS limit (`byte_idx < (n + 7) // 8`) | yes (both triggers) |
| haiku | dictionary_decode | triton | C | `pid * BLOCK_SIZE` is int32 (`shl.b32 %r3, %ctaid.x, 8`, then `cvt.s64.s32`): n = 2, grid 2^23 + 1 programs; program 2^23 has `idx = -2^31 + i`, which passes `idx < n`, and reads `codes[-2^31]`. The refused cache-hint load is this `codes` load | yes: read 8 GiB below `codes`, block 8388608 |
| haiku | merge_sorted | numba | limit | MC FL: the loop reads `a[i]` only under `i < m` and `b[j]` only under `j < p`; each step advances one cursor, so `i + j = idx` after the loop, and `i >= m` gives `j = idx - m < p`. Sortedness is not needed | no |
| haiku | merge_sorted | nvcc | limit | MC FL, as Numba (the index is a 32-bit unsigned sum, zero-extended, so it stays in [0, 2^32) and is compared with m + p) | no |
| haiku | min_max | nvcc | limit | DS: the reduction keeps 2 * 256 doubles in dynamic shared memory, which the CUDA C++ front end does not give (0 bytes); with 4096 bytes every access is in bounds. The model's `test.cu` never launches the kernel | yes: invalid shared writes from 0x400 |
| haiku | radix_scatter | numba | C | as Fable: grid (1, 2, 1) stores `out[1]`. Also E: `cursors[0] = 1` on entry | yes (both triggers) |
| haiku | radix_scatter | nvcc | C | as Numba | yes (both triggers) |
| haiku | radix_scatter | triton | C | int32 program offset: program 2^23 reads `keys[-2^31]`. The `out` store is also unguarded: with 2 programs in y, or `cursors[0]` out of range on entry, it writes past `out` | yes: read 8 GiB below `keys` |
| haiku | rolling_sum | nvcc | limit | LR: `for (i = start; i <= idx; i++)` with `start >= 0` and `idx < n` | no |
| haiku | rolling_sum | triton | C | int32 program offset: for program 2^23, `range(start, pos + 1)` is empty, so no load runs, but `out[-2^31 + i]` is stored. The refused cache-hint store is this store | yes: write 16 GiB below `out` |
| haiku | run_end_decode | numba | E | r = 0 with n = 1 (a column with rows needs r >= 1, but the claim gives r the minimum 0): `right = -1`, the loop is skipped, and `run_values[0]` of the empty buffer is read. For r >= 1, `left` stays in [0, r - 1] for any `run_ends` | yes: invalid read at 0x0 (the empty buffer) |
| haiku | run_end_decode | nvcc | E | as Numba | yes: invalid read at 0x0 |
| haiku | sorted_contains | numba | limit | BS | no |
| haiku | sorted_contains | nvcc | limit | BS | no |
| haiku | string_contains | numba | limit | LR SZ DL: `chars[start + i + j]` with `i <= str_len - k` and `j < k` lies below `end` | no |
| haiku | string_contains | nvcc | limit | LR SZ DL; the int32 sums `start + i + j <= end - 1` do not overflow | no |
| haiku | string_contains | triton | C | int32 program offset: program 2^23 reads `offsets[-2^31]`, then stores `match[-2^31 + i]` (the refused cache-hint store) | yes: read 8 GiB below `offsets` |
| haiku | valid_rows | numba | C | the store is guarded by `pos < rows.size` only. Count zero on entry, n = 2^31 - 1 rows all valid, grid (2^23, 2, 1): 2^32 - 2 increments of the int32 count; the atomic returns -2^31, which passes the guard, and Numba's wraparound stores `rows[-1]`. Also E with a 1-D grid: `count[0] = -2` on entry, n = 1, stores `rows[-1]` | yes: the guard element before `rows` was overwritten with a row id (0.5 s); the E trigger with memcheck |
| haiku | valid_rows | nvcc | C | unguarded store: grid (1, 2, 1) stores `rows[1]`. Also E: `count[0] = 1` on entry. The refused load is the QS limit | yes (both triggers) |
| haiku | valid_rows | triton | C | int32 program offset: program 2^23 reads `valid[-2^28]` (`row_id // 8` of a negative row). The `rows` store is also unguarded (y grid, or `count` on entry) | yes: read 256 MiB below `valid` |
| opus | merge_sorted | numba | C | a grid with more blocks than the tiles: m = p = 1, a = [1], b = [2] (sorted), grid 2. In block 1, k0 = 1792 > m + p, so na = lo1 - lo0 = 1 - 1791 = -1790, nb = 0, and the merge loop takes `bi < nb` with bi = -1790 and reads `buf[na + bi]` = `buf[-3580]`, which Numba's wraparound makes `buf[-1788]`, 14 KiB below `buf`. The refused load is `a[mid]` in the split search (BS; in bounds) | yes: invalid shared read at 0xffffc820, block 1 |
| opus | merge_sorted | nvcc | C | as Numba: `buf[-3580]`. The refused `call.uni` is the `__ldg(a + mid)` of `split`, in bounds (BS) | yes: invalid shared read at 0xffff9020, block 1 |
| opus | merge_sorted | triton | limit | KS PL RD BM. Every memory access is masked to a range the merge-path search keeps valid, for any contents of `a` and `b`: `_merge_splits` keeps each `lo` in `[max(k - p, 0), min(k, m)]`, so `i0, i1` are in `[0, m]` and `j0 = k0 - i0` in `[0, p]`; `value_a = a[i0 + t]` is masked by `t < na = i1 - i0`, so `i0 + t < i1 <= m`; `value_b = b[j0 + (BLOCK - 1 - t)]` is masked by `t >= BLOCK - nb`, so `BLOCK - 1 - t < nb` and `j0 + nb = k1 - i1 <= p`; the store `out[k0 + t]` is masked by `t < k1 - k0 <= BLOCK`. The bitonic merge (BM) is register only | yes: unsorted `b`, m = p = 4096, its own grid, no memcheck error |
| opus | radix_scatter | numba | C | cursors meet the contract: n = 1, keys = [0], `cursors[0] = 0`, the others 1; grid (1, 2, 1): both blocks reserve a slot of byte 0 and the second stores `out[1]`. Also E with a 1-D grid (a cursor out of range on entry). The refused load `keys[tile_start + x]` with `x < min(n - tile_start, TILE)` is in bounds (MM) | yes |
| opus | radix_scatter | nvcc | C | as Numba. The refused `call.uni` is `__ldg(keys + tile_start + x)`, in bounds (MM) | yes |
| opus | radix_scatter | triton | C | cursors meet the contract: n = 2, keys = [0, 0], `cursors[0] = 0`, the others 2; grid (1, 2): the two programs share `program_id(0) = 0`, both `atomic_add` on `cursors` and gather `cursor - tile_start`, and the second writes `out[2]`/`out[3]`. Also E with one program: `cursors[0] = 2` on entry writes `out[2]`/`out[3]`. The refused `st.shared.v4.b32` is the `tl.sort` shared store (SW), a limit, but the `out` overrun is the defect | yes (both triggers): out overrun in block (0,1,0), and single-program entry |
| opus | rolling_sum | numba | limit | CD. Every index is below its array for `window` in [1, 1024]: `span = 1023 + w <= 2047 < 2048`, the suffix pass reads `xs[x]` for `x < suffix_end <= 1022 + w`, the prefix pass `xs[4t + j + w - 1] <= 2046`, step 3 `xs[u + w - 1] <= 2046`, `prefix_carry[u // 4]`, `suffix_carry[u // 8]`, and `out[i]` under `i < n` | no |
| opus | rolling_sum | nvcc | limit | LR. In bounds as in Numba. At level 1 nvcc keeps the loops rolled and walks each with a pointer or index next to its counter: the staging loop tests the 32-bit `x < span` but addresses `values` and `xs` through `g` and pointers that advance with it, and the chunk loops of steps 1a, 1b, and 3 do the same; those registers have no fact at the loop heads. The values load is `__ldg(values + g)` under `g >= 0 && g < n` | no |
| opus | rolling_sum | triton | limit | PL RD. Reads `values[g]` under `(g >= 0) & (g < n)` (a masked load, PL) and stores `out[g_store]` under `(t_store >= halo) & (g_store < n)`, with `g_store >= pid * (REGION - halo) >= 0`. The offset is 64-bit (`mul.lo.s64` of `2049 - w` by `ctaid`). The `associative_scan`, `_row` reduction (RD), and `gather` are register operations. No defect | no |
| opus | run_end_decode | numba | E | the last run end is not n: r = 1, n = 2, `run_ends = [1]`, grid 1. `first_run = 0`; row 1 has `ends[0] = 1 <= 1`, so a = 1 and it reads `run_values[first_run + 1]` = `run_values[1]`. With strictly increasing run ends and `run_ends[r - 1] = n`, `a` is at most the run of the row and `first_run + a <= r - 1`. r = 0 with n >= 1 reads `run_values[0]` as well. The refused load `run_ends[probe]` with `probe < hi <= r - 1` is in bounds (KS) | yes: invalid read at `run_values + 8` |
| opus | run_end_decode | nvcc | E | as Numba. The refused `call.uni` is `__ldg(run_ends + probe)`, in bounds (KS) | yes: invalid read at `run_values + 8` |
| opus | run_end_decode | triton | E | the last run end is not n: r = 1, n = 2, `run_ends = [1]`, grid 1. `j0 = 0`; run 0 has `rel_end = run_ends[0] - 0 = 1`, so `starts[1] = 1` and `cumsum(starts)[1] = 1`; row 1 (`in_range`) reads `run_values[j0 + 1] = run_values[1]`. With strictly increasing run ends and `run_ends[r - 1] = n`, `j0 + cumsum(starts) <= r - 1`. The refused masked load `run_ends[pos]` in the k-ary search (KS) is in bounds | yes: invalid read at `run_values + 8` |
| opus | sorted_contains | numba | limit | BS in its base-and-length form: `pos = pos + half if members[pos + half] <= key else pos; length -= half` keeps `pos + length <= s` with `length >= 1`, so `pos + half < s` and `members[pos]` has `pos <= s - 1`; `members[0]` is read only when `s > 0` | no |
| opus | sorted_contains | nvcc | limit | BS, as Numba (grid-stride loop over `i < n`). The refused `call.uni` is `__ldg(members + pos + half)`; without `__ldg` the gate refuses the same two loads with the same address forms (`variants/sc_plain`) | no |
| opus | sorted_contains | triton | limit | BS. `base` starts at 0, grows by `half = span // 2` only where `members[base + half] <= key`, and `span` shrinks so `base + span <= s` stays; the two masked loads `members[base + half]` and `members[base]` are then below s. The diagnosis shows both loads with no bound on `base` | no (diag) |
| opus | string_contains | numba | limit | SQ LA LR SZ DL. `long_queue[atomic.add(long_count, 0, 1)]`: `long_count` is set to 0 before a barrier and each of the 256 threads adds 1 at most once, so the slot is below 256; the queued values are thread ids of rows below n, so `offsets[s0 + long_queue[q]]` and `match[ls]` are in bounds; the scans read `chars[p]` only for `p <= end - k` and `chars[p + j]` with `j < k` | no |
| opus | string_contains | nvcc | limit | SQ LA LR SZ DL, as Numba; `head[u]` is written in a loop that nvcc does not unroll at level 1 | no |
| opus | string_contains | triton | limit | PL LR SZ DL FL. The lane scan reads `text[p] = chars[start + p]` under `p < length = offsets[s+1] - offsets[s]` and, in the long-string loop, `long_text[q + t]` under `q <= last_start = length - k` and `t < k`; `offsets[s]`, `offsets[s+1]` under `s < n`; `pattern[j]` under `j < k`; the store `match[s]` under `s < n`. The register-word matching is arithmetic | no |
| opus | valid_rows | numba | C | count zero on entry, n = 1, valid = [1], grid (1, 2, 1): both blocks reserve a slot and the second stores `rows[1]`. Also E with a 1-D grid: `count[0] = 1` on entry. The refused load `valid[row0 >> 3]` is the QS limit | yes |
| opus | valid_rows | nvcc | C | as Numba. The refused `call.uni` is `__ldg(valid + (row0 >> 3))`, in bounds under the contract (QS) | yes |
| opus | valid_rows | triton | C | count zero on entry, n = 2, valid = [3], grid (1, 2): both programs share `program_id(0) = 0`, both `atomic_add` on `count`, and the second stores `rows[2]`/`rows[3]`. Also E with one program: `count[0]` out of range on entry. The refused masked load `valid[pid * BYTES + local_byte]` is the QS limit; the defect is the unguarded `rows` store, which the diagnosis lists (`rows + count_position`, the position unbounded) | yes (both triggers) |
| sonnet | days_to_year | triton | C | `pid * BLOCK_SIZE` is int32: n = 2, grid 2^23 + 1 programs; program 2^23 has offsets -2^31 + t, which pass `offs < n`, and reads `days[-2^31 + t]` | yes: read 8 GiB below `days`, block 8388608 |
| sonnet | dictionary_decode | triton | C | int32 offsets, as `days_to_year`: program 2^23 reads `codes[-2^31 + t]` | yes: read 8 GiB below `codes` |
| sonnet | merge_sorted | numba | limit | BS: `i = (lo + hi + 1) // 2` lies in (lo, hi] with `lo >= max(0, k - p)` and `hi <= min(k, m)`, so `a[i - 1]` and `b[j]` (read only when `j < p`) are in bounds; after the loop `i >= m` gives `j = k - m < p`. Sortedness is not needed | no |
| sonnet | merge_sorted | nvcc | limit | BS, as Numba | no |
| sonnet | min_max | triton | C | int32 offsets: program 2^23 reads `values[-2^31 + t]` | yes: read 16 GiB below `values` |
| sonnet | radix_scatter | numba | C | as Fable: grid (1, 2, 1) stores `out[1]`. Also E: `cursors[0] = 1` on entry | yes (both triggers) |
| sonnet | radix_scatter | nvcc | C | as Numba | yes (both triggers) |
| sonnet | radix_scatter | triton | C | int32 offsets: program 2^23 reads `keys[-2^31 + t]`; its `out` store is also unguarded, as in Numba | yes: read 8 GiB below `keys` |
| sonnet | rolling_sum | numba | limit | DS: the tile is dynamic shared memory of (255 + window) * 8 bytes, at most 10232, which the Numba front end does not give; under its own `run`, which passes those bytes, every access is in bounds | yes: invalid shared writes from 0x400 |
| sonnet | rolling_sum | nvcc | limit | DS, as Numba (`extern __shared__ double tile[]`); the load `values[g]` under `g >= 0 && g < n`, the refused instruction, is in bounds | yes: invalid shared writes from 0x400 |
| sonnet | rolling_sum | triton | C | int32 rows: the loads are masked by `idx >= 0`, but program 2^23 stores `out[-2^31 + t]` | yes: write 16 GiB below `out` |
| sonnet | run_end_decode | numba | E | r = 1, n = 2, `run_ends = [1]`: row 1 ends the search with `lo = 1 = r` and reads `run_values[1]`. With `run_ends[r - 1] = n > i`, lo <= r - 1 | yes: invalid read at `run_values + 8` |
| sonnet | run_end_decode | nvcc | E | as Numba | yes: invalid read at `run_values + 8` |
| sonnet | sorted_contains | numba | limit | BS | no |
| sonnet | sorted_contains | nvcc | limit | BS | no |
| sonnet | string_contains | numba | limit | DS: the pattern is staged in dynamic shared memory of k bytes, which no constant covers; the `chars` load needs LR SZ DL | yes: invalid shared writes from 0x400 |
| sonnet | string_contains | nvcc | limit | DS, as Numba (`extern __shared__`) | yes: invalid shared writes from 0x400 |
| sonnet | string_contains | triton | C | one program per string and no `pid < n` test: n = 2 strings, k = 2, grid 3; program 2 reads `offsets[3]` and then stores `match[2]` | yes: invalid read at `offsets + 12`, block 2 |
| sonnet | valid_rows | numba | C | unguarded store: grid (1, 2, 1) stores `rows[1]`. Also E: `count[0] = 1` on entry. The refused load is the QS limit | yes (both triggers) |
| sonnet | valid_rows | nvcc | C | as Numba | yes (both triggers) |
| sonnet | valid_rows | triton | C | int32 rows: program 2^23 of a grid of 2^23 + 1 reads `valid[-2^28]` (`row // 8` of row -2^31); the `rows` store is also unguarded | yes: read 256 MiB below `valid` |

New feature codes, with the count of limits that need each; the limits also need LR (11), SZ (9), DL (9), FL (4), RD (3), and PL (3):

| Code | Construct | Limits |
| --- | --- | --- |
| BS | a binary search: the midpoint `(lo + hi) >> 1`, `(lo + hi) / 2`, `(lo + hi + 1) // 2`, or a base-and-length form `pos + half`, must be proved between `lo` and `hi`, and `lo`, `hi`, updated by `selp`, must stay in `[0, size]` across the loop; the checker keeps `lo < hi` but not the midpoint's place between them | 12 |
| DS | dynamic shared memory in Numba or CUDA C++ (`cuda.shared.array(0)`, `extern __shared__`), which the front ends do not give: `(255 + window) * 8` bytes, `k` bytes, or `2 * 256` doubles | 5 |
| KS | a k-ary search: each lane probes `lo + t * stride` with a register stride, a product of two registers | 2 |
| MM | bounds through `min` and `max` of sizes and block offsets, `h = max(0, t0 - window + 1)`, `e = t0 + tile - h` | 2 |
| MC | merge cursors: a sequential merge advances `i` or `j` by one per step, so `i + j = k`, an equality of three registers | 2 |
| SQ | a shared work queue: a count zeroed before a barrier and incremented at most once by each thread; the gate reads shared memory as any value | 2 |
| LA | a local array written at a loop index that the compiler does not unroll; a local store is accepted only at a constant offset of a local variable | 2 |
| BL | binary lifting, `j = where(cond, j + step, j)` with a halving `step >= 1`: the select loses `j >= 0` | 1 |
| CD | Numba's count-down `range(K - 1, -1, -1)`, lowered as `j = -c` with an up-counter `c`: the invariant `j = 1 - c` holds on entry only after substituting `c`'s constant, which the check does not do | 1 |
| BM | a bitonic merge in registers between the masked loads and the store; it does no addressing but carries the lengths that the masks use | 1 |
| QS | a quotient length: the bitmap's `(n + 7) / 8` bytes are a size symbol of their own, so `valid[i >> 3]` with `i < n` is not provable | 0 (the refused instruction of 10 `valid_rows` defects) |
| CP | a Triton store with a cache-policy operand, whose stored value the translation drops, so a store to an index output fails its value check | 0 (Haiku's `radix_scatter`, a defect) |

| Model | C | E | limit | Refused |
| --- | --- | --- | --- | --- |
| fable | 6 | 5 | 9 | 20 |
| haiku | 9 | 2 | 8 | 19 |
| opus | 8 | 3 | 10 | 21 |
| sonnet | 11 | 2 | 8 | 21 |
| all | 34 | 12 | 35 | 81 |

Classes of defect: a counter or cursor that the kernel trusts to stay below its output's length (C, 20: every refused `valid_rows` and `radix_scatter` kernel but Haiku's and Sonnet's Triton ones); 32-bit Triton program offsets (C, 11: Haiku's 5 Triton kernels and Sonnet's 6 but `string_contains`); a merge tile past the output for a grid larger than the data (C, 2: Opus `merge_sorted`, Numba and CUDA C++); a missing program guard (C, 1: Sonnet `string_contains` Triton); the run-end relation `run_ends[r - 1] = n`, or `r >= 1` when `n >= 1` (E, 10: every refused `run_end_decode`); sortedness (E, 2: Fable `merge_sorted`, Numba and CUDA C++).

The accepted held-out kernels, all safe under their bindings by `verify_text_sound`, give their NumPy references through the executor (`tests/executor/test_corpus_launch.py`).
