"""HandTap2PDScore web app: Gradio UI at / and a JSON REST endpoint for your own front end.

    python app/app.py                  # http://localhost:7860
    curl -F video=@clip.mp4 http://localhost:7860/api/predict

On a Hugging Face Space (sdk: gradio) this file is the entry point; the Space runs
`python app.py`, which serves on port 7860.
"""
import json
import os
import sys
import tempfile
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
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from handtap.predict import HandTapPredictor, plot_result  # noqa: E402

# Default = the exact training extraction (predictions verified identical, see
# outputs/results/inference_check.csv). HANDTAP_FAST=1 (960 px, backends in parallel) is ~40 %
# faster but changed 1 of 12 UPDRS scores and 1 of 12 PD labels in that check.
FAST = os.environ.get("HANDTAP_FAST", "0") == "1"
PREDICTOR = HandTapPredictor(threads=int(os.environ.get("HANDTAP_THREADS", os.cpu_count() or 2)),
                             **(dict(max_side=960, det_every=1, parallel=True) if FAST else {}))
MAX_SECONDS = float(os.environ.get("HANDTAP_MAX_SECONDS", 30))  # MDS-UPDRS needs ~10 taps
MAX_MB = 200

# ----------------------------------------------------------------------------- REST API
api = FastAPI(title="HandTap2PDScore API", version="0.1.0")
api.add_middleware(CORSMiddleware, allow_origins=os.environ.get("HANDTAP_CORS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


@api.get("/api/health")
def health():
    return dict(status="ok", backends=PREDICTOR.backends,
                models={t: dict(featureset=b["featureset"], model=b["model"], cv_metrics=b["cv_metrics"])
                        for t, b in PREDICTOR.bundles.items()})


@api.post("/api/predict")
def predict_endpoint(video: UploadFile = File(...), signals: bool = True):
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
        return PREDICTOR.predict(path, max_seconds=MAX_SECONDS, keep_signals=signals)
    except IOError as e:
        raise HTTPException(400, str(e))
    finally:
        os.unlink(path)


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


def analyse(video, progress=gr.Progress()):
    if not video:
        raise gr.Error("Upload a finger-tapping video first.")
    progress(0, desc="tracking hand")
    r = PREDICTOR.predict(video, max_seconds=MAX_SECONDS,
                          progress=lambda x: progress(0.95 * x, desc="tracking hand"))
    progress(0.97, desc="scoring")
    kin = pd.DataFrame([dict(metric=v["label"], value=v["value"], unit=v["unit"], meaning=v["description"])
                        for v in r["kinematics"].values()])
    taps = pd.DataFrame(r["taps"])
    if len(taps):
        taps = taps[["tap", "t_peak", "duration", "amplitude", "open_speed", "close_speed"]].round(3)
    fig = plot_result(r)
    out = Path(tempfile.mkdtemp()) / (Path(r["video"]).stem + "_handtap.json")
    out.write_text(json.dumps(r, indent=2))
    return _summary_md(r), fig, kin, taps, str(out)


with gr.Blocks(title="HandTap2PDScore") as demo:
    gr.Markdown("# HandTap2PDScore\nFinger-tapping video → hand keypoints (MediaPipe + RTMPose) → "
                "kinematics → MDS-UPDRS 3.4 score and Parkinson's probability.\n\n"
                "Film one hand from the side, whole hand in view, tapping index finger on thumb "
                "as fast and wide as possible for 10–20 s.")
    with gr.Row():
        with gr.Column(scale=1):
            vid = gr.Video(label="Finger-tapping video", sources=["upload", "webcam"])
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column(scale=2):
            summary = gr.Markdown()
            plot = gr.Plot(label="Aperture and detected taps")
    with gr.Tab("Kinematic metrics"):
        kin = gr.Dataframe(wrap=True)
    with gr.Tab("Per-tap table"):
        taps = gr.Dataframe()
    js = gr.File(label="Full result (JSON)")
    btn.click(analyse, vid, [summary, plot, kin, taps, js], api_name="analyse")

app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
