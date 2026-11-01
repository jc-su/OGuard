# radix_scatter Implementation

## Approach
Radix scatter using atomic operations. For each row, extract the byte at position shift, then atomically advance the cursor for that byte and write the row ID.

## Status
✓ Completed

## Implementation Details
- Extracts byte: (key >> shift) & 0xFF
- Uses atomic add to allocate positions within each byte bucket
- Supports shifts 0, 8, 16, 24 (for 32-bit radix sort)
- Cursors array maintained as mutable input/output
