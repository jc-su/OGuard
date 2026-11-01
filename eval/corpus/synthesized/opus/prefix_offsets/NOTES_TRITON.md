# prefix_offsets (Triton)

The kernel is complete and passes `test_triton.py`.

## Scan across programs

A single-pass scan over several programs (decoupled look-back) needs a status
array that is zero before the launch, so that a program can tell whether its
predecessor has published its tile sum. The task's signature has no scratch
argument, and `offsets` holds arbitrary values on entry, so no such array is
available. A second launch is not possible either, because `run` launches the
one kernel with the task's arguments.

Each program therefore handles a contiguous range of whole tiles and first sums
all rows before its range itself, then scans its range tile by tile. The
programs do not communicate, so the result is correct for any grid size. The
cost is redundant reading: with P programs the kernel reads about
n * P / 2 lengths in the first phase. `run` caps P at 32 (`PROGRAMS`).

Measured on the RTX A6000 at n = 1000003 (GPU shared with other jobs, so
approximate): 83 us with 32 programs, against about 430 us for a single
program that scans the whole column.

## Overflow

The task bounds each length by 2^20 but not the total, so for large n the
int32 offsets wrap. The kernel adds in int32, which wraps modulo 2^32 in any
order, so the result equals the sequential int32 scan. The test compares
against an int64 cumulative sum cast to int32 and includes inputs that wrap.
