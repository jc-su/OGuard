"""The CUDA C++ front end: nvcc at NVVM optimization level 1.

`nvcc_operator` compiles CUDA C++ source to PTX with `nvcc -ptx -O3 -Xcicc
-O1`: NVVM at level 1, as for Numba, so that a loop keeps its bound test
(design section 5.9). The kernel must be declared `extern "C"`, so that its
entry has the source's name. nvcc passes each pointer and each scalar as one
parameter (`Convention.C`).

Imports `opgate.signature`, `opgate.binding`, and `opgate.frontends`; runs
`nvcc`.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from opgate.binding import GRID_LIMIT, NO_SHARED, Convention, validate_grid, validate_shared
from opgate.frontends import Operator, Target
from opgate.signature import Affine, Signature

__all__ = ["nvcc_operator"]


def nvcc_operator(
    source: str,
    kernel: str,
    signature: Signature,
    block: tuple[int, int, int],
    target: Target,
    *,
    grid: tuple[int, int, int] = GRID_LIMIT,
    shared: Affine = NO_SHARED,
) -> Operator:
    """The operator of the `extern "C"` kernel `kernel` of the CUDA C++ `source`,
    launched in blocks of `block` threads and grids of at most `grid` blocks, any
    grid of the GPU unless the claim states less."""
    validate_grid(grid)
    validate_shared(signature.sizes, shared)
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is not installed")
    with tempfile.TemporaryDirectory() as tmp:
        path, ptx = Path(tmp) / "kernel.cu", Path(tmp) / "kernel.ptx"
        path.write_text(source)
        command = [nvcc, "-ptx", f"-arch={target.arch}", "-O3", "-Xcicc", "-O1"]
        result = subprocess.run(
            [*command, str(path), "-o", str(ptx)], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise ValueError(f"nvcc refused the source:\n{result.stderr}")
        text = ptx.read_text()
    return Operator(text, kernel, signature, Convention.C, block, grid, shared)
