import numpy as np


def parse_node_mask(mask_str: str, N: int) -> np.ndarray:
    # "1|0|0|1|0"
    parts = mask_str.split("|")
    if len(parts) != N:
        raise ValueError(f"node_mask has {len(parts)} entries, expected {N}")
    return np.array([float(p) for p in parts], dtype=np.float32)


def route_mask(route: str, note: int, N: int) -> np.ndarray:
    route = route.lower()
    m = np.zeros((N,), dtype=np.float32)
    if route == "all":
        m[:] = 1.0
        return m
    i = int(note) % N
    if route == "one":
        m[i] = 1.0
        return m
    if route == "two":
        m[i] = 1.0
        m[(i + 2) % N] = 1.0
        return m
    raise ValueError(f"unknown route: {route}")
