# Raw regression fixtures

These are field projections of the historical records named by each file.
All retained PTX, bindings, requests, responses, output bytes, completion times,
and observations are unchanged. JSON whitespace is compacted and gzip is used
only for storage. No checksums or derived identities are added.

The projections omit machine-local archive and proof-library paths (including
`provenance.e3_archive` and `environment.formal_library`). Tests supply a local
library marker when exercising path validation. These fixtures are historical
regression data, not performance or safety measurements of the current source.
The complete experiment records remain in `eval/records` in the development
repository. The archive fixtures run no GPU kernels.
