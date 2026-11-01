# rolling_sum Implementation

## Approach
Naive rolling sum where each thread computes the sum over its window. Thread i sums from max(0, i - window + 1) to i.

## Status
✓ Completed

## Implementation Details
- Each thread independently computes its rolling sum
- Window size from 1 to 1024
- Handles boundary at start of array correctly
- No inter-thread synchronization needed
