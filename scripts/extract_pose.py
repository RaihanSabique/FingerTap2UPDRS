"""Stage 1: hand keypoints for every dataset video, all backends in one decode pass.

    python scripts/extract_pose.py --shard 0 --n_shards 40

Writes outputs/pose/<ID>.npz (keys: t, fps, size, box, box_score, <backend>_kpts,
<backend>_score, <backend>_valid, mediapipe_world). Existing files are skipped.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handtap import ROOT  # noqa: E402
from handtap.pose import BACKENDS, extract_keypoints  # noqa: E402

VIDEOS = ROOT / "data/videos_FIS/videos"
OUT = ROOT / "outputs/pose"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    vids = sorted(v for v in VIDEOS.iterdir() if v.suffix.lower() in (".mp4", ".mov", ".avi", ".m4v"))
    vids = vids[args.shard::args.n_shards]
    for v in vids:
        dst = OUT / f"{v.stem}.npz"
        if dst.exists():
            continue
        t0 = time.time()
        r = extract_keypoints(v, backends=args.backends, threads=args.threads)
        flat = {k: r[k] for k in ("t", "fps", "size", "box", "box_score")}
        for b in args.backends:
            for k, a in r[b].items():
                flat[f"{b}_{k}"] = a
        np.savez_compressed(dst.with_suffix(".tmp.npz"), **flat)
        dst.with_suffix(".tmp.npz").rename(dst)
        cov = {b: round(float(r[b]["valid"].mean()), 3) for b in args.backends}
        print(f"{v.stem}: {len(r['t'])} frames in {time.time() - t0:.0f}s, coverage {cov}", flush=True)
