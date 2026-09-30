"""Download the pretrained hand-pose weights into models/ and verify their SHA-256.

    python scripts/download_models.py             # MediaPipe + RTMDet-hand + RTMPose-hand (app, ~67 MB)
    python scripts/download_models.py --vitpose   # + ViTPose-B wholebody (research comparison, 360 MB)

The trained classifiers (models/classifier/*.joblib) are part of the repository.
"""
import argparse
import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MMPOSE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
WEIGHTS = {
    # dst: (url, member inside the zip or None, sha256, licence)
    "mediapipe/hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        None, "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1", "Apache-2.0"),
    "rtmdet_hand/rtmdet_nano_hand.onnx": (
        MMPOSE + "rtmdet_nano_8xb32-300e_hand-267f9c8f.zip", "end2end.onnx",
        "568d3ea97a5b142488366b67e036b6a5cb0a1fef9087a710cb8e66b6979fbac2", "Apache-2.0"),
    "rtmpose_hand/rtmpose_m_hand5_256.onnx": (
        MMPOSE + "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip", "end2end.onnx",
        "39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29", "Apache-2.0"),
}
VITPOSE = {
    "vitpose_wholebody/vitpose-b-wholebody.onnx": (
        "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-b-wholebody.onnx",
        None, "017c08e161bfa2c72a96297b3b6de975ecd83c65c4fd56fb5b0bfc1298709069", "Apache-2.0"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dst, url, member, digest):
    path = ROOT / "models" / dst
    if path.exists() and sha256(path) == digest:
        print(f"ok       {dst}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {dst}  <-  {url}")
    with urllib.request.urlopen(url) as r:
        data = r.read()
    if member:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = next(n for n in z.namelist() if n.endswith(member))
            with z.open(name) as src, open(path, "wb") as out:
                shutil.copyfileobj(src, out)
    else:
        path.write_bytes(data)
    got = sha256(path)
    if got != digest:
        sys.exit(f"checksum mismatch for {dst}: {got} (expected {digest}); upstream file changed")
    print(f"ok       {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vitpose", action="store_true", help="also fetch the ViTPose-B wholebody ONNX")
    args = ap.parse_args()
    todo = dict(WEIGHTS, **(VITPOSE if args.vitpose else {}))
    for dst, (url, member, digest, _) in todo.items():
        fetch(dst, url, member, digest)
