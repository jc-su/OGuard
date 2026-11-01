# E3: triggering the corpus defects on the GPU

Recorded 2026-09-26. This file records experiment E3 of the OpGate design (section 1.1). For each of the 27 refused corpus kernels that `classification.md` marks as a defect, it builds the classified trigger, runs the model's unchanged kernel on GPU 1 of the build box, and shows that the defect is a real out-of-bounds access.

## Method

The harness makes the effect observable without a tool. It puts the kernel's buffers in one pooled allocation, as a GPU engine's pool allocator does. It places a canary of a known value immediately before and after each output, and a victim buffer that holds a secret immediately after each input. A changed canary is an out-of-bounds write into another tenant's buffer. A committed output that changes when only the secret changes is an out-of-bounds read of another tenant's data. For read defects the harness runs the trigger twice with two different secrets and compares the committed output; it runs this differential test only for read defects, because a write defect with atomic output ordering gives a nondeterministic output that would look like a read.

Each trigger also runs with every buffer as its own allocation, to show whether it faults instead of corrupting a neighbor, and under compute-sanitizer memcheck in both layouts.

One process handles one (kernel, layout) run, because a memory fault ends the CUDA context for the whole process. The Numba kernels run through `cuda.jit(kernel)` with the model's `numba.py` imported by path (never on `sys.path`, because the file would shadow the numba package). The CUDA C++ kernels compile to a cubin with `nvcc -cubin -arch=sm_86 -O3`, which loads and launches from the same Python pool through the driver API. Every binary is deleted right after it loads.

Toolchain: Python 3.12 (conda), Numba 0.65 built-in CUDA target, NVVM default level, PyTorch 2.11+cu130 not used for the kernels, nvcc CUDA 13.0.88, sm_86. compute-sanitizer is CUDA 12.9 (version 2025.2.1.0); it runs a CUDA 13 binary correctly (a check kernel reports the right result and 0 errors), so it is used for the CUDA 13 kernels. GPU 1 only (`CUDA_VISIBLE_DEVICES=1`), an RTX A6000 with 48 GB.

Sizes. Most triggers use small buffers: n = 64 for filter_select, capacity 2048 with 1000 probes for the hash_probe capacity bound, capacity 16 with one probe for the counter-on-entry hash_probe, n = 1000 with 4 blocks of 256 for the no-row-count kernels, and n = 2 with C = 10 for the offsets-relation string read. The three signed-wrap kernels need a full-size input: the values or chars buffer is allocated to `offsets[g]` = 2^31 - 1 elements on the device (16 GiB of float64 for segmented_sum, 2 GiB of uint8 for string_length), never on the host. The segment starts at `offsets[0]` = 2^31 - 2 so that `start + lane` overflows int32 for the high lanes after one in-bounds read, which reaches the same wrap as `offsets = [0, 2^31 - 1]` without a 67-million-iteration warm-up.

Verdict codes are from `classification.md`: `C` is reachable under the task contract; `E` is reachable when a precondition the catalog cannot check is violated.

## Results

| Model | Task | Toolchain | Verdict | Trigger used | Pool effect | compute-sanitizer (pool) | compute-sanitizer (separate) | Separate, no tool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| haiku | filter_select | numba | E | count[0]=-(n+1)=-65; one match; idx=-65 passes idx<n; Numba negative-index wrap -> sel[-1] | silent cross-buffer write | 0 errors | 2 errors: invalid write 4B | no fault |
| haiku | filter_select | nvcc | E | count[0]=-1; one match; idx=-1 passes idx<n; sel[-1] | silent cross-buffer write | 0 errors | 2 errors: invalid write 4B | no fault |
| sonnet | filter_select | numba | E | count[0]=n=64; one match; pos=n; no guard; sel[n] | silent cross-buffer write | 0 errors | 2 errors: invalid write 4B | no fault |
| sonnet | filter_select | nvcc | E | count[0]=n=64; one match; pos=n; sel[n] | silent cross-buffer write | 0 errors | 2 errors: invalid write 4B | no fault |
| fable | filter_select | numba | E | count[0]=n=64; one match; base=n, rank=0; sel[n] | silent cross-buffer write | 0 errors | 2 errors: invalid write 4B | no fault |
| fable | filter_select | nvcc | E | count[0]=n=64; one match; base=n, rank=0; sel[n] | silent cross-buffer write | 0 errors | 2 errors: invalid write 4B | no fault |
| haiku | hash_probe | numba | C | capacity=2048, out_cap=100, 1000 matches; idx up to 999 passes idx<capacity; out_build[100..999] | silent cross-buffer write | 0 errors | 32 errors: invalid write 4B | no fault |
| haiku | hash_probe | nvcc | C | capacity=2048, out_cap=100, 1000 matches; idx up to 999 passes idx<capacity; out_build[100..999] | silent cross-buffer write | 0 errors | 32 errors: invalid write 4B | no fault |
| sonnet | hash_probe | numba | E | out_count[0]=-101; one match; pos wraps (Numba) to out_build[-1] | silent cross-buffer write | 0 errors | 5 errors: invalid write 4B | no fault |
| sonnet | hash_probe | nvcc | E | out_count[0]=-1; one match; pos=-1; pos<out_cap; out_build[-1] | silent cross-buffer write | 0 errors | 5 errors: invalid write 4B | no fault |
| fable | hash_probe | numba | E | out_count[0]=-101; one match; pos wraps (Numba) to out_build[-1] | silent cross-buffer write | 0 errors | 5 errors: invalid write 4B | no fault |
| fable | hash_probe | nvcc | E | out_count[0]=-1; one match; pos=-1; pos<out_cap; out_build[-1] | silent cross-buffer write | 0 errors | 5 errors: invalid write 4B | no fault |
| haiku | scatter_by_offsets | numba | E | flags[5]=1 with offsets[5]=total=8; out[total] (out length total) | silent cross-buffer write | 0 errors | 2 errors: invalid write 4B | no fault |
| haiku | scatter_by_offsets | nvcc | E | flags[5]=1 with offsets[5]=total=8; out[total] (out length total) | silent cross-buffer write | 0 errors | 2 errors: invalid write 4B | no fault |
| fable | scatter_by_offsets | numba | E | flags[5]=1 with offsets[5]=total=8; out[total] (out length total) | silent cross-buffer write | 0 errors | 2 errors: invalid write 4B | no fault |
| fable | scatter_by_offsets | nvcc | E | flags[5]=1 with offsets[5]=total=8; out[total] (out length total) | silent cross-buffer write | 0 errors | 2 errors: invalid write 4B | no fault |
| haiku | string_length | numba | C | lengths bound int32[1] (4 bytes); model TYPES int64[:] does an 8-byte store; high 4 bytes land past the column | silent cross-buffer write | 0 errors | 2 errors: invalid write 8B | no fault |
| haiku | groupby_sum | nvcc | C | no row count; guard i<gridDim*blockDim; 4 blocks over 1000 rows read group/value[1000..1023] | silent cross-buffer read | 0 errors | 25 errors: invalid read 4B | no fault |
| haiku | hash_build | nvcc | C | no row count; 4 blocks over 1000 rows read keys[1000..1023] | silent cross-buffer read | 0 errors | 25 errors: invalid read 8B | no fault |
| haiku | histogram | nvcc | C | no row count; 4 blocks over 1000 rows read keys[1000..1023]; counts[bin] reveals the secret keys | silent cross-buffer read | 0 errors | 25 errors: invalid read 4B | no fault |
| haiku | q6_sum | nvcc | C | n=gridDim*blockDim replaces row count; 4 blocks over 1000 rows read shipdate/discount/quantity/price[1000..1023] | silent cross-buffer read | 0 errors | 26 errors: invalid read 4B | no fault |
| haiku | starts_with | nvcc | C | n=gridDim*blockDim-1=1023; reads offsets[1001..1023] and writes match[1000..1022] | silent cross-buffer write | 0 errors | 23 errors: invalid read 4B | no fault |
| haiku | string_length | nvcc | C | n=gridDim*blockDim-1=1023; reads offsets[1001..1023] and writes lengths[1000..1022] | silent cross-buffer write | 0 errors | 24 errors: invalid read 4B | no fault |
| fable | starts_with | numba | E | n=2, offsets=[10,0,10], k=5; uint32 length offsets[1]-begin wraps for i=0; reads chars[10..14] (past C=10) | silent cross-buffer read | 0 errors | 2 errors: invalid read 1B | no fault |
| sonnet | segmented_sum | nvcc | C | g=1, offsets=[2147483646,2147483647] (buffer 16 GiB float64); int stride overflows near INT_MAX; wrapped index read | fault | 32 errors: invalid read 8B | 32 errors: invalid read 8B | fault |
| fable | string_length | nvcc | C | g=1, offsets=[2147483646,2147483647] (buffer 2 GiB uint8); int stride overflows near INT_MAX; wrapped index read | fault | 31 errors: invalid read 1B | 31 errors: invalid read 1B | fault |
| fable | segmented_sum | nvcc | C | g=1, offsets=[2147483646,2147483647] (buffer 16 GiB float64); int stride overflows near INT_MAX; wrapped index read | fault | 32 errors: invalid read 8B | 32 errors: invalid read 8B | fault |

## Counts

- Defects triggered: 27 of 27. None came back with no effect, so every classified defect is a real out-of-bounds access.
- Silent cross-buffer write in the pool layout: 19.
- Silent cross-buffer read in the pool layout: 5.
- Fault (illegal address) in the pool layout: 3 (the three signed 32-bit wrap kernels).

- compute-sanitizer reports 0 errors in the pool layout for 24 of 27 defects. These are the 24 that stay inside the one pool allocation. The three wrap kernels are the exception, because their wrapped index is about 16 GiB (or 2 GiB) from the buffer and lands outside the pool.
- compute-sanitizer reports an error in the separate-allocation layout for 27 of 27 defects.
- With separate allocations and no tool, 24 of 27 defects do not fault. A one-element or few-element overrun lands in the allocator's padding, so only the sanitizer catches it in that layout. This is why the pool blind spot matters: in a shared pool the same access hits a live neighbor and is silent to both the process and the sanitizer.

By verdict: 12 `C` and 15 `E`, matching `classification.md`.

## Notes and anything that contradicts the classification

Nothing contradicts the classification. Every one of the 27 defects triggered. The following points refine the result:

- The three signed 32-bit wrap kernels (sonnet segmented_sum, fable segmented_sum, fable string_length, all nvcc) fault with an illegal address (CUDA error 700) rather than corrupting a neighbor. The overflow of `start + lane` sends the load about 16 GiB (float64) or 2 GiB (uint8) before the buffer, which no pooled canary can catch, so both layouts fault and compute-sanitizer reports the invalid read in both. This is the availability threat of the design (scenario 1): one such kernel ends the context for every tenant.
- haiku starts_with (nvcc) and haiku string_length (nvcc) both read offsets past the end and write match or lengths past the end, because the kernel takes the row count from `gridDim.x * blockDim.x - 1`. The pooled observable is the write into the trailing canary; the separate-layout sanitizer surfaces the offsets read first. Both accesses are out of bounds.
- The pool must be at least as large as the overrun for the write to stay silent. The haiku hash_probe capacity-bound defect writes out_build[100..999] (3.6 KB past a 400-byte output), and haiku string_length (nvcc) writes lengths[1000..1022] (92 bytes past the column). With a trailing canary smaller than the overrun, the write leaves the pool and the sanitizer catches it even in the pool layout, so the trigger sizes the trailing tenant buffer to contain the overrun, which a multi-GB engine pool always does. The haiku string_length (numba) int64 store writes only 4 bytes past the int32 column, so it stays silent with any canary.
- haiku hash_probe (nvcc) has a second defect that the classification notes: out_count = -1 on entry writes out_build[-1]. This run triggers the capacity-bound defect (the `C` verdict). The out_count = -1 variant is the same mechanism as the sonnet and fable counter-on-entry hash_probe defects, which are triggered here.
- The counter-on-entry write defects (filter_select and hash_probe) land one element before or after the output. In the pool this changes the adjacent canary; with separate allocations it lands in allocator padding and does not fault, so only the sanitizer flags it.
