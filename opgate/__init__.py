"""OpGate: untrusted GPU operators in a shared analytics engine, isolated by proof.

The package is layered. Imports flow downward only:

    signature   the author's claim about each argument; imports nothing here
    binding     the symbolic statement of an operator's launches, in size symbols
    catalog     tenants, buffers, grants, statuses, the step check and launch check
    gate        PTX text and a binding to a decision, through Rocq
    executor    the only execute path: step check, gate, module loading, launch
    frontends   PTX text, a signature, and a calling convention per toolchain
    hosts       engines the executor is installed into

The terms are defined in docs/GLOSSARY.md and enforced by tests/test_vocabulary.py.
"""
