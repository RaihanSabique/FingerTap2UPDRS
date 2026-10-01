"""HandTap2PDScore web app: Gradio UI at / and a JSON REST endpoint for your own front end.

    python app/app.py                  # http://localhost:7860
    curl -F video=@clip.mp4 http://localhost:7860/api/predict

Password protection: set HANDTAP_PASSWORD (and optionally HANDTAP_USER, default "handtap").
The Gradio page then shows a login form and the REST endpoints need HTTP Basic auth:
    curl -u handtap:<password> -F video=@clip.mp4 https://<space>.hf.space/api/predict
On a Hugging Face Space (SPACE_ID set) the app refuses to start without a password.

On a Hugging Face Space (sdk: gradio) this file is the entry point; the Space runs
`python app.py`, which serves on port 7860.
"""
import json
import os
import re
import secrets
import sys
import tempfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
# works both in the repo (app/ next to handtap/) and in a flat Space folder
for p in (HERE, HERE.parent):
    if (p / "handtap").is_dir():
        sys.path.insert(0, str(p))
        os.environ.setdefault("HANDTAP_MODELS", str(p / "models"))
        break

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.security import HTTPBasic, HTTPBasicCredentials  # noqa: E402

from handtap import available_cpus  # noqa: E402
from handtap.overlay import render_overlay  # noqa: E402
from handtap.predict import HandTapPredictor, plot_result  # noqa: E402

# Default = the exact training extraction (predictions verified identical, see
# outputs/results/inference_check.csv). HANDTAP_FAST=1 (960 px, backends in parallel) is ~40 %
# faster but changed 1 of 12 UPDRS scores and 1 of 12 PD labels in that check.
FAST = os.environ.get("HANDTAP_FAST", "0") == "1"
# MediaPipe cross-check off by default: the classifiers use RTMPose features only, so it
# does not change predictions but doubles processing time. The overlay video is the
# visual quality check instead.
QC_DEFAULT = os.environ.get("HANDTAP_QC", "0") == "1"
# threads from the container's real CPU quota, not os.cpu_count() (the host's count)
THREADS = int(os.environ.get("HANDTAP_THREADS", 0)) or available_cpus()
PREDICTOR = HandTapPredictor(threads=THREADS, qc=QC_DEFAULT,
                             **(dict(max_side=960, det_every=1, parallel=True) if FAST else {}))
OVERLAY_DIR = Path(tempfile.gettempdir()) / "handtap_overlays"
OVERLAY_DIR.mkdir(exist_ok=True)
OVERLAY_TTL = int(os.environ.get("HANDTAP_OVERLAY_TTL", 3600))   # seconds a rendered video is kept
OVERLAY_WIDTH = int(os.environ.get("HANDTAP_OVERLAY_WIDTH", 960))
MAX_SECONDS = float(os.environ.get("HANDTAP_MAX_SECONDS", 30))  # MDS-UPDRS needs ~10 taps
MAX_MB = 200

# ----------------------------------------------------------------------------- auth
AUTH_USER = os.environ.get("HANDTAP_USER", "handtap")
AUTH_PASSWORD = os.environ.get("HANDTAP_PASSWORD", "")
if not AUTH_PASSWORD and os.environ.get("SPACE_ID"):
    # fail closed: a Space without the secret would otherwise be open to anyone
    raise RuntimeError("HANDTAP_PASSWORD is not set: add it under Space settings -> Variables and secrets")
_basic = HTTPBasic(auto_error=False)


def require_auth(cred: HTTPBasicCredentials | None = Depends(_basic)):
    """HTTP Basic check for the REST API (no-op when no password is configured)."""
    if not AUTH_PASSWORD:
        return
    ok = (cred is not None
          and secrets.compare_digest(cred.username.encode(), AUTH_USER.encode())
          and secrets.compare_digest(cred.password.encode(), AUTH_PASSWORD.encode()))
    if not ok:
        raise HTTPException(401, "authentication required", headers={"WWW-Authenticate": "Basic"})


# ----------------------------------------------------------------------------- overlay
def _purge_overlays():
    """Delete rendered overlays older than OVERLAY_TTL (they contain the uploaded hand video)."""
    now = time.time()
    for f in OVERLAY_DIR.glob("*.mp4"):
        try:
            if now - f.stat().st_mtime > OVERLAY_TTL:
                f.unlink()
        except OSError:
            pass


def make_overlay(video_path, res, det, progress=None):
    """Render the tracking overlay for one prediction; returns the MP4 path."""
    b = det["primary"]
    raw = det["raw"][b]
    u, pdp = res["prediction"].get("updrs", {}), res["prediction"].get("pd", {})
    probs = [u.get("probabilities", {}).get(str(c), 0.0) or 0.0 for c in range(4)]
    info = dict(
        title=Path(res["video"]).stem[:32],
        subtitle=f"{res['duration_s']:.1f} s · {res['frames']} frames · RTMPose-m hand tracking",
        updrs_line=(f"model {u['score']}  (ordinal score {u['expected_score']:.2f})" if u else ""),
        probs=probs if u else None,
        pd_line=(f"P(Parkinson's) = {pdp['probability']:.2f}" if pdp else ""),
        footer="Bars: share of trees voting each score. Research prototype, not a diagnostic device.")
    out = OVERLAY_DIR / f"{uuid.uuid4().hex}.mp4"
    return render_overlay(video_path, raw["kpts"], raw["valid"], det["sigs"][b], det["taps"][b],
                          det["features"][b], info, out, width=OVERLAY_WIDTH, stride=2,
                          preset="veryfast", crf=30, progress=progress)


# ----------------------------------------------------------------------------- REST API
api = FastAPI(title="HandTap2PDScore API", version="0.1.0")
api.add_middleware(CORSMiddleware, allow_origins=os.environ.get("HANDTAP_CORS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


@api.get("/api/health", dependencies=[Depends(require_auth)])
def health():
    return dict(status="ok", backends=PREDICTOR.backends,
                models={t: dict(featureset=b["featureset"], model=b["model"], cv_metrics=b["cv_metrics"])
                        for t, b in PREDICTOR.bundles.items()})


@api.post("/api/predict", dependencies=[Depends(require_auth)])
def predict_endpoint(video: UploadFile = File(...), signals: bool = True, overlay: bool = False,
                     qc: bool = QC_DEFAULT):
    """overlay=true also renders the tracking video; fetch it from `overlay_url` (same auth)
    within `overlay_expires_s`. qc=true adds the MediaPipe cross-check (about 2x slower)."""
    _purge_overlays()
    suffix = Path(video.filename or "clip.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        n = 0
        while chunk := video.file.read(1 << 20):
            n += len(chunk)
            if n > MAX_MB << 20:
                raise HTTPException(413, f"video larger than {MAX_MB} MB")
            f.write(chunk)
        path = f.name
    try:
        res, det = PREDICTOR.predict(path, max_seconds=MAX_SECONDS, keep_signals=signals, qc=qc,
                                     return_details=True)
        if overlay:
            t0 = time.time()
            mp4 = make_overlay(path, res, det)
            res["overlay_url"] = f"/api/overlay/{mp4.name}"
            res["overlay_expires_s"] = OVERLAY_TTL
            res["overlay_render_s"] = round(time.time() - t0, 1)
        return res
    except IOError as e:
        if str(e).startswith("cannot open video"):
            raise HTTPException(400, str(e))   # bad upload; anything else is a server fault
        raise
    finally:
        os.unlink(path)


@api.get("/api/overlay/{name}", dependencies=[Depends(require_auth)])
def overlay_file(name: str):
    if not re.fullmatch(r"[0-9a-f]{32}\.mp4", name) or not (OVERLAY_DIR / name).exists():
        raise HTTPException(404, "overlay not found or expired")
    return FileResponse(OVERLAY_DIR / name, media_type="video/mp4")


# ----------------------------------------------------------------------------- Gradio UI
def _summary_md(r):
    p = r["prediction"]
    lines = []
    if "updrs" in p:
        u = p["updrs"]
        lines.append(f"## MDS-UPDRS 3.4 finger tapping: **{u['score']}**  "
                     f"<small>(expected value {u['expected_score']})</small>")
    if "pd" in p:
        lines.append(f"### P(Parkinson's disease) = **{p['pd']['probability']:.2f}** → {p['pd']['label']}")
    q = r["quality"]
    if q["warnings"]:
        lines.append("\n**Quality warnings**\n" + "\n".join(f"- ⚠️ {w}" for w in q["warnings"]))
    else:
        lines.append(f"\nTracking quality OK (backend agreement r = {q.get('backend_agreement_r')})")
    if "updrs" in p and p["updrs"]["cv_metrics"]:
        m = p["updrs"]["cv_metrics"]
        lines.append(f"\n<small>UPDRS model ({p['updrs']['featureset']}/{p['updrs']['model']}) "
                     f"cross-validated on unseen participants: MCC {m.get('MCC')}, "
                     f"accuracy {m.get('Accuracy')}, A-AC {m.get('A_AC')}.</small>")
    if "pd" in p and p["pd"]["cv_metrics"]:
        m = p["pd"]["cv_metrics"]
        lines.append(f"<small>PD model: MCC {m.get('MCC')}, sensitivity {m.get('Sensitivity')}, "
                     f"specificity {m.get('Specificity')}, AUC {m.get('ROC_AUC')}.</small>")
    lines.append(f"\n<small>{r['disclaimer']} Processed {r['frames']} frames in {r['processing_s']} s.</small>")
    return "\n".join(lines)


def analyse(video, want_overlay=True, qc=False, progress=gr.Progress()):
    if not video:
        raise gr.Error("Upload a finger-tapping video first.")
    _purge_overlays()
    track_share = 0.7 if want_overlay else 0.95
    progress(0, desc="tracking hand")
    r, det = PREDICTOR.predict(video, max_seconds=MAX_SECONDS, qc=qc, return_details=True,
                               progress=lambda x: progress(track_share * x, desc="tracking hand"))
    overlay = None
    if want_overlay:
        overlay = str(make_overlay(video, r, det, progress=lambda x: progress(
            track_share + (0.97 - track_share) * x, desc="rendering tracking video")))
    progress(0.98, desc="scoring")
    kin = pd.DataFrame([dict(metric=v["label"], value=v["value"], unit=v["unit"], meaning=v["description"])
                        for v in r["kinematics"].values()])
    taps = pd.DataFrame(r["taps"])
    if len(taps):
        taps = taps[["tap", "t_peak", "duration", "amplitude", "open_speed", "close_speed"]].round(3)
    fig = plot_result(r)
    out = Path(tempfile.mkdtemp()) / (Path(r["video"]).stem + "_handtap.json")
    out.write_text(json.dumps(r, indent=2))
    return _summary_md(r), overlay, fig, kin, taps, str(out)


with gr.Blocks(title="HandTap2PDScore", delete_cache=(1800, OVERLAY_TTL)) as demo:
    gr.Markdown("# HandTap2PDScore\nFinger-tapping video → hand keypoints (MediaPipe + RTMPose) → "
                "kinematics → MDS-UPDRS 3.4 score and Parkinson's probability.\n\n"
                "Film one hand from the side, whole hand in view, tapping index finger on thumb "
                "as fast and wide as possible for 10–20 s.")
    with gr.Row():
        with gr.Column(scale=1):
            vid = gr.Video(label="Finger-tapping video", sources=["upload", "webcam"])
            want_overlay = gr.Checkbox(value=True, label="Tracking overlay video",
                                       info="hand skeleton + live aperture trace (adds ~10-20 s)")
            qc = gr.Checkbox(value=QC_DEFAULT, label="MediaPipe cross-check (slower)",
                             info="second hand tracker; reports tracking agreement and mm amplitude")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column(scale=2):
            summary = gr.Markdown()
            overlay = gr.Video(label="Tracking overlay (RTMPose-m hand)", autoplay=True)
            plot = gr.Plot(label="Aperture and detected taps")
    with gr.Tab("Kinematic metrics"):
        kin = gr.Dataframe(wrap=True)
    with gr.Tab("Per-tap table"):
        taps = gr.Dataframe()
    js = gr.File(label="Full result (JSON)")
    btn.click(analyse, [vid, want_overlay, qc], [summary, overlay, plot, kin, taps, js], api_name="analyse")

# ssr_mode=False: on Spaces Gradio otherwise starts its own Node SSR server on port 7860,
# which collides with the uvicorn server below
app = gr.mount_gradio_app(api, demo, path="/", ssr_mode=False,
                          auth=(AUTH_USER, AUTH_PASSWORD) if AUTH_PASSWORD else None,
                          auth_message="HandTap2PDScore research prototype. Sign in to continue.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
