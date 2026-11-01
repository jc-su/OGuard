# min_max Implementation

## Approach
Block-wise reduction using shared memory and atomic operations. Each block performs a tree reduction to find its local min/max, then atomics update the global result.

## Status
✓ Completed

## Implementation Details
- Each block does a cooperative reduction on shared memory
- Uses atomic min/max to update global result
- Handles n=0 case (result stays as initialized: +inf, -inf)
- CUDA version uses atomic CAS for double min/max operations
