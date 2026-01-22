import csv
import os
import wave
import numpy as np
from typing import Dict, List


def write_wav_stereo(path: str, sr: int, left: np.ndarray, right: np.ndarray) -> None:
    assert left.shape == right.shape
    os.makedirs(os.path.dirname(path), exist_ok=True)
    audio = np.stack([left, right], axis=1)
    audio_i16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())


def read_note_csv(path: str) -> List[Dict]:
    events: List[Dict] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            events.append({
                "time": float(row["time_sec"]),
                "type": row["type"].strip(),
                "note": int(row["note"]),
                "vel": float(row.get("vel", "0.0") or 0.0),
                "route": (row.get("route") or "").strip() or None,
                "node_mask": (row.get("node_mask") or "").strip() or None,
            })
    events.sort(key=lambda e: e["time"])
    return events
