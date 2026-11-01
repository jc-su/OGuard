# merge_sorted Implementation

## Approach
Each thread uses binary search to determine which elements from a and b should appear at its output position, then outputs the correct element. Thread at position idx performs binary search to find how many elements from a come before it.

## Status
✓ Completed

## Implementation Details
- Each thread independently determines its output value
- Binary search on split point ensures correct ordering
- Preserves order for equal elements (a before b)
- Work-efficient for GPU with thousands of threads
