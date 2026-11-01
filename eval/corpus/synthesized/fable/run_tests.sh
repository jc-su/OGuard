#!/usr/bin/env bash
# Runs the Numba test (which also writes numba.ptx) and the CUDA test for each task directory.
# Usage: ./run_tests.sh [task ...]   (no arguments: every task directory)
set -u
export CUDA_VISIBLE_DEVICES=0
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ "$#" -eq 0 ]; then
    set -- filter_select gather projection q6_sum hash_build hash_probe groupby_sum \
           scatter_by_offsets string_length starts_with histogram segmented_sum
fi
status=0
for task in "$@"; do
    dir="$HERE/$task"
    echo "== $task: numba"
    (cd "$dir" && python -P test.py 2>&1 | grep -v -i "NumbaPerformanceWarning\|warn(") || status=1
    echo "== $task: cuda"
    (cd "$dir" && nvcc -ptx -arch=sm_86 -O3 kernel.cu -o kernel.ptx \
        && nvcc -arch=sm_86 -O3 kernel.cu test.cu -o test && ./test) || status=1
done
exit $status
