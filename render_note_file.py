import argparse
import json
import os
import numpy as np

from modal_synth.engine import Engine, EngineConfig, ArticulationParams, NodeNetworkParams
from modal_synth.io_utils import read_note_csv, write_wav_stereo
from modal_synth.routing import parse_node_mask, route_mask
from modal_synth.metrics import basic_metrics


def render_csv_to_wav(
    note_csv: str,
    out_wav: str,
    manifest_json: str,
    sr: int = 48000,
    N: int = 5,
    K: int = 4,
    voice_count: int = 16,
    default_route: str = "all",
    tail_sec: float = 2.0,
):
    events = read_note_csv(note_csv)
    if not events:
        raise ValueError("No events in CSV")

    duration = max(e["time"] for e in events) + tail_sec
    total = int(duration * sr)

    cfg = EngineConfig(sample_rate=sr, N=N, K=K, voice_count=voice_count)
    engine = Engine(cfg, art=ArticulationParams(), body=NodeNetworkParams())

    out = np.zeros((total,), dtype=np.float32)

    evt_i = 0
    for n in range(total):
        t = n / sr
        while evt_i < len(events) and events[evt_i]["time"] <= t:
            e = events[evt_i]
            if e["type"] == "on":
                if e["node_mask"]:
                    mask = parse_node_mask(e["node_mask"], N)
                else:
                    route = e["route"] or default_route
                    mask = route_mask(route, e["note"], N)
                engine.note_on(e["note"], e["vel"], node_mask=mask)
            else:
                engine.note_off(e["note"])
            evt_i += 1

        out[n] = engine.step_one_sample()

    # simple trim
    out *= 0.2
    L, R = out.copy(), out.copy()

    write_wav_stereo(out_wav, sr, L, R)

    m = basic_metrics(L)
    os.makedirs(os.path.dirname(manifest_json) or ".", exist_ok=True)
    with open(manifest_json, "w") as f:
        json.dump({
            "note_csv": note_csv,
            "out_wav": out_wav,
            "sr": sr,
            "N": N,
            "K": K,
            "voice_count": voice_count,
            "default_route": default_route,
            "duration_sec": duration,
            "metrics": m,
        }, f, indent=2)

    print(f"Wrote {out_wav}")
    print("Metrics:", m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note_csv", required=True)
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--manifest_json", required=True)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--voices", type=int, default=16)
    ap.add_argument("--default_route", default="all", choices=["one", "two", "all"])
    ap.add_argument("--tail_sec", type=float, default=2.0)
    args = ap.parse_args()

    render_csv_to_wav(
        note_csv=args.note_csv,
        out_wav=args.out_wav,
        manifest_json=args.manifest_json,
        sr=args.sr,
        N=args.N,
        K=args.K,
        voice_count=args.voices,
        default_route=args.default_route,
        tail_sec=args.tail_sec,
    )


if __name__ == "__main__":
    main()
