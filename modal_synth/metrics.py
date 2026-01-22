import numpy as np
from typing import Dict


def basic_metrics(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"peak": 0.0, "rms": 0.0, "dc": 0.0, "crest": 0.0}

    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))
    dc = float(np.mean(x))
    crest = float(peak / (rms + 1e-12))
    return {"peak": peak, "rms": rms, "dc": dc, "crest": crest}
