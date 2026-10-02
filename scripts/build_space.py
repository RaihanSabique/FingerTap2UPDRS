"""Stage 6: assemble a self-contained deploy folder (Hugging Face Space or any Docker host).

    python scripts/build_space.py            # -> deploy/hf_space/

Contents: app.py, the handtap package, the CPU weights it needs (MediaPipe hand
landmarker, RTMDet-nano hand, RTMPose-m hand, the two classifier bundles), pinned
requirements, a Space README (YAML header), a Dockerfile and .gitattributes for LFS.
ViTPose weights are not shipped: the app does not use them.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "deploy/hf_space"

PKG_FILES = ["__init__.py", "video.py", "pose.py", "kinematics.py", "models.py", "predict.py", "overlay.py"]
WEIGHTS = ["mediapipe/hand_landmarker.task", "rtmdet_hand/rtmdet_nano_hand.onnx",
           "rtmpose_hand/rtmpose_m_hand5_256.onnx",
           "classifier/handtap_updrs.joblib", "classifier/handtap_pd.joblib"]
PINNED = ["numpy", "scipy", "pandas", "scikit-learn", "xgboost", "joblib", "opencv-python-headless",
          "mediapipe", "onnxruntime", "matplotlib", "pillow", "imageio-ffmpeg", "gradio", "fastapi", "uvicorn", "python-multipart"]

README = """---
title: FingerTap2UPDRS
emoji: 🖐️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: {gradio}
python_version: "3.10"
app_file: app.py
pinned: false
license: other
short_description: Finger-tapping video to MDS-UPDRS 3.4 score and PD risk
---

# HandTap2PDScore

Upload a finger-tapping video; the app tracks the hand with **MediaPipe HandLandmarker** and
**RTMDet + RTMPose-m (hand5)**, extracts tapping kinematics (rate, amplitude, decrement,
hesitations, rhythm, smoothness) and predicts the **MDS-UPDRS 3.4** finger-tapping score (0-3)
and the probability of Parkinson's disease.

## Access

The app is public. To require a login, set the Space secret `HANDTAP_PASSWORD` (and
optionally `HANDTAP_USER`, default `EmoryViTAL`); the UI then shows a login form and the API
needs HTTP Basic auth with the same credentials.

## API (for a custom front end)

```
POST /api/predict      multipart form field `video`  -> JSON (prediction, kinematics, taps, signals, quality)
                       ?overlay=true  also render the tracking video (download from overlay_url, kept 1 h)
                       ?qc=true       add the MediaPipe cross-check (slower)
GET  /api/health       loaded models and their cross-validated metrics
```

```js
const fd = new FormData(); fd.append("video", file);
const res = await fetch("https://<user>-<space>.hf.space/api/predict", {{method: "POST", body: fd}});
const result = await res.json();   // result.prediction.updrs.score, result.prediction.pd.probability, ...
```

Environment variables: `HANDTAP_MAX_SECONDS` (default 30), `HANDTAP_THREADS`, `HANDTAP_FAST` (0; 1 = ~2x faster, less exact), `HANDTAP_CORS`
(comma-separated allowed origins, default `*`).

**Research prototype.** Trained on 234 clips / 118 participants from one Spanish hospital
(HUBU-FIS, Zenodo 17738775). Not a medical device; not validated on other cameras or populations.
Model weights: MediaPipe (Apache-2.0), RTMPose/RTMDet (OpenMMLab, Apache-2.0).
"""

DOCKERFILE = """FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
"""


def pinned():
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    have = {l.split("==")[0].lower(): l for l in out.splitlines() if "==" in l}
    return [have.get(p.lower(), p) for p in PINNED]


if __name__ == "__main__":
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "handtap").mkdir(parents=True)
    for f in PKG_FILES:
        shutil.copy(ROOT / "handtap" / f, DST / "handtap" / f)
    shutil.copy(ROOT / "app/app.py", DST / "app.py")
    for w in WEIGHTS:
        (DST / "models" / w).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "models" / w, DST / "models" / w)
    req = pinned()
    (DST / "requirements.txt").write_text("\n".join(r for r in req if not r.startswith("gradio")) + "\n")
    gv = next((r.split("==")[1] for r in req if r.startswith("gradio==")), "5.0.0")
    (DST / "README.md").write_text(README.format(gradio=gv))
    (DST / "Dockerfile").write_text(DOCKERFILE)
    # apt packages for the HF gradio image: opencv-contrib needs libGL; mediapipe dlopens libEGL/GLES even on CPU
    (DST / "packages.txt").write_text("libgl1\nlibglib2.0-0\nlibegl1\nlibgles2\n")
    (DST / "requirements-docker.txt").write_text("\n".join(req) + "\n")
    (DST / ".gitattributes").write_text("*.onnx filter=lfs diff=lfs merge=lfs -text\n"
                                        "*.task filter=lfs diff=lfs merge=lfs -text\n"
                                        "*.joblib filter=lfs diff=lfs merge=lfs -text\n")
    size = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file()) / 1e6
    print(f"wrote {DST} ({size:.0f} MB)")
