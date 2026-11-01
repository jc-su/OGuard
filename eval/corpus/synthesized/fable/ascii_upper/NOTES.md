# ascii_upper

- The work is one item per byte, and the byte count is `offsets[n]`, which is
  not a kernel argument. The kernel reads `offsets[n]` itself and strides over
  the grid, so any launch of at least one block is correct. `run` sizes the
  grid from `chars.size`, which equals `offsets[n]` by the column layout.
