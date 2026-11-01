import triton
import triton.language as tl

TYPES = {
    "p_partkey_ptr": "*i64",
    "p_brand_id_ptr": "*i32",
    "p_container_grp_ptr": "*i32",
    "p_size_ptr": "*i32",
    "brand_map_ptr": "*i32",
    "container_map_ptr": "*i32",
    "size_map_ptr": "*i32",
    "map_size": "i32",
    "n_parts": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    p_partkey_ptr,
    p_brand_id_ptr,
    p_container_grp_ptr,
    p_size_ptr,
    brand_map_ptr,
    container_map_ptr,
    size_map_ptr,
    map_size,
    n_parts,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_parts

    key = tl.load(p_partkey_ptr + offs, mask=mask, other=-1)
    valid = mask & (key >= 0) & (key < map_size)

    brand_id = tl.load(p_brand_id_ptr + offs, mask=mask, other=0)
    container_grp = tl.load(p_container_grp_ptr + offs, mask=mask, other=0)
    p_size = tl.load(p_size_ptr + offs, mask=mask, other=0)

    tl.store(brand_map_ptr + key, brand_id, mask=valid)
    tl.store(container_map_ptr + key, container_grp, mask=valid)
    tl.store(size_map_ptr + key, p_size, mask=valid)
