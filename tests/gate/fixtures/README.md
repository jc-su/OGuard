# Historical proof-regression inputs

These PTX files are copied byte for byte from the archived experiment artifacts
named by each directory. The slow original-kernel proof regressions can use them
without the full experiment records. The two Triton sorted-contains inputs are
already byte-identical in the packaged corpus and those tests read those copies.
The artifact's documented corpus-path anonymization applies to those copies.
No original experiment records or compiler instructions are changed.

These are static checker inputs. Modified boundary controls in the tests are not
GPU execution workloads. No hashes or identity fields are added.
