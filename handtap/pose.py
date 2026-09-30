"""Hand keypoint backends. Every backend returns the same 21-joint hand layout
(MediaPipe / COCO-WholeBody-hand order: 0 wrist, 1-4 thumb CMC..tip, 5-8 index MCP..tip,
9-12 middle, 13-16 ring, 17-20 pinky) in full-frame pixel coordinates.

    mediapipe : Google MediaPipe HandLandmarker (palm detector + landmark model, VIDEO tracking)
    rtmpose   : RTMDet-nano hand detector + RTMPose-m hand5 (SimCC, 256x256), OpenMMLab ONNX
    vitpose   : RTMDet-nano hand box + ViTPose-B COCO-WholeBody (133 kpts), hand block only

Only CPU onnxruntime / mediapipe are needed, so the same code runs in a HF Space.
"""
import cv2
import numpy as np

from . import MODELS

N_KP = 21
THUMB_TIP, INDEX_TIP, WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 4, 8, 0, 5, 9, 17


def _ort_session(path, threads=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def box_affine(box, out_w, out_h, pad=1.25):
    """Affine mapping an (x0,y0,x1,y1) box, padded and aspect-fixed, onto an out_w x out_h crop."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * pad, (y1 - y0) * pad
    aspect = out_w / out_h
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect
    s = out_w / w
    return np.array([[s, 0, out_w / 2 - s * cx], [0, s, out_h / 2 - s * cy]], np.float32)


def invert_points(M, pts):
    Mi = cv2.invertAffineTransform(M)
    return pts @ Mi[:, :2].T + Mi[:, 2]


# ----------------------------------------------------------------------------- RTMDet hand
class HandDetector:
    """RTMDet-nano trained on hands (OpenMMLab). Input 320x320 letterbox, BGR, pad 114."""
    SIZE = 320
    MEAN = np.array([103.53, 116.28, 123.675], np.float32)
    STD = np.array([57.375, 57.12, 58.395], np.float32)

    def __init__(self, threads=None):
        self.sess = _ort_session(MODELS / "rtmdet_hand/rtmdet_nano_hand.onnx", threads)

    def __call__(self, img):
        h, w = img.shape[:2]
        s = self.SIZE / max(h, w)
        rs = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.SIZE, self.SIZE, 3), 114, np.uint8)
        canvas[:rs.shape[0], :rs.shape[1]] = rs
        x = ((canvas.astype(np.float32) - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        dets, _ = self.sess.run(None, {"input": x})
        dets = dets[0]
        if len(dets) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        return dets[:, :4] / s, dets[:, 4]


class BoxTracker:
    """Keeps one hand box per frame: the most confident detection, preferring the one that
    overlaps the previous box, then an exponential smoother so crops don't jitter."""

    def __init__(self, min_score=0.3, alpha=0.6):
        self.prev, self.min_score, self.alpha = None, min_score, alpha

    @staticmethod
    def _iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0

    def update(self, boxes, scores):
        keep = scores >= self.min_score
        boxes, scores = boxes[keep], scores[keep]
        if len(boxes) == 0:
            return self.prev, 0.0
        if self.prev is None:
            # tapping hand is the largest confident one in these close-ups
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            j = int(np.argmax(scores * np.sqrt(area)))
        else:
            j = int(np.argmax(scores + np.array([self._iou(b, self.prev) for b in boxes])))
        b = boxes[j].astype(np.float32)
        self.prev = b if self.prev is None else self.alpha * b + (1 - self.alpha) * self.prev
        return self.prev.copy(), float(scores[j])


# ----------------------------------------------------------------------------- RTMPose hand
class RTMPoseHand:
    """RTMPose-m hand5 (COCO-WholeBody-hand, OneHand10K, FreiHand, RHD, Halpe-hand), SimCC."""
    SIZE = 256
    MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    STD = np.array([58.395, 57.12, 57.375], np.float32)

    def __init__(self, threads=None):
        self.sess = _ort_session(MODELS / "rtmpose_hand/rtmpose_m_hand5_256.onnx", threads)

    def __call__(self, img, box):
        M = box_affine(box, self.SIZE, self.SIZE, pad=1.25)
        crop = cv2.warpAffine(img, M, (self.SIZE, self.SIZE), flags=cv2.INTER_LINEAR)
        rgb = crop[:, :, ::-1].astype(np.float32)          # mmpose trained with to_rgb=True
        x = ((rgb - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        sx, sy = self.sess.run(None, {"input": x})
        xi, yi = sx[0].argmax(-1), sy[0].argmax(-1)
        pts = np.stack([xi, yi], -1).astype(np.float32) / 2.0  # simcc_split_ratio = 2
        score = np.minimum(sx[0].max(-1), sy[0].max(-1))
        return invert_points(M, pts), score


# ----------------------------------------------------------------------------- ViTPose wholebody
class ViTPoseWholebodyHand:
    """ViTPose-B COCO-WholeBody (easy_ViTPose ONNX, 256x192 input, 64x48 heatmaps).
    Returns whichever of its two hand blocks (91-111 left, 112-132 right) is more confident."""
    W, H = 192, 256
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, threads=None):
        self.sess = _ort_session(MODELS / "vitpose_wholebody/vitpose-b-wholebody.onnx", threads)

    def __call__(self, img, box):
        M = box_affine(box, self.W, self.H, pad=1.4)
        crop = cv2.warpAffine(img, M, (self.W, self.H), flags=cv2.INTER_LINEAR)
        x = ((crop[:, :, ::-1].astype(np.float32) / 255 - self.MEAN) / self.STD)
        hm = self.sess.run(None, {"input_0": x.transpose(2, 0, 1)[None]})[0][0]
        K, h, w = hm.shape
        flat = hm.reshape(K, -1)
        idx = flat.argmax(1)
        score = flat.max(1)
        py, px = (idx // w).astype(np.float32), (idx % w).astype(np.float32)
        # quarter-pixel shift toward the higher neighbour (standard heatmap refinement)
        for k in range(K):
            ix, iy = int(px[k]), int(py[k])
            if 0 < ix < w - 1:
                px[k] += 0.25 * np.sign(hm[k, iy, ix + 1] - hm[k, iy, ix - 1])
            if 0 < iy < h - 1:
                py[k] += 0.25 * np.sign(hm[k, iy + 1, ix] - hm[k, iy - 1, ix])
        pts = np.stack([px * self.W / w, py * self.H / h], -1)
        left, right = slice(91, 112), slice(112, 133)
        blk = left if score[left].mean() >= score[right].mean() else right
        return invert_points(M, pts[blk]), score[blk]


# ----------------------------------------------------------------------------- MediaPipe
class MediaPipeHand:
    def __init__(self, fps=30.0):
        from mediapipe.tasks.python import BaseOptions, vision
        self.vision = vision
        self.lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "mediapipe/hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3))
        self.last_ms = -1

    def __call__(self, img, t, ref_box=None):
        import mediapipe as mp
        H, W = img.shape[:2]
        ms = max(int(round(t * 1000)), self.last_ms + 1)
        self.last_ms = ms
        r = self.lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img[:, :, ::-1])), ms)
        if not r.hand_landmarks:
            return None, None, None
        cand = [np.array([[p.x * W, p.y * H] for p in h], np.float32) for h in r.hand_landmarks]
        if ref_box is not None:
            c = np.array([(ref_box[0] + ref_box[2]) / 2, (ref_box[1] + ref_box[3]) / 2])
            j = int(np.argmin([np.linalg.norm(k.mean(0) - c) for k in cand]))
        else:
            j = int(np.argmax([np.ptp(k[:, 0]) * np.ptp(k[:, 1]) for k in cand]))
        world = np.array([[p.x, p.y, p.z] for p in r.hand_world_landmarks[j]], np.float32)
        score = np.full(N_KP, r.handedness[j][0].score, np.float32)
        return cand[j], score, world

    def close(self):
        self.lm.close()


BACKENDS = ("mediapipe", "rtmpose", "vitpose")


def _box_from_kpts(k, score, pad=1.15, min_score=0.4):
    """Hand box from the previous frame's RTMPose keypoints (tracking between detections)."""
    if k is None or not np.isfinite(k).all() or np.mean(score) < min_score:
        return None
    x0, y0 = k.min(0)
    x1, y1 = k.max(0)
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * pad, (y1 - y0) * pad
    side = max(w, h)
    return np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], np.float32)


def extract_keypoints(video_path, backends=BACKENDS, max_seconds=None, threads=None, progress=None,
                      max_side=None, det_every=1, parallel=False):
    """Run the chosen backends over a video in one decode pass.

    Returns {"t", "fps", "size", "box", "box_score", <backend>: {"kpts","score","valid"[,"world"]}}.
    Keypoints are always in original-frame pixels. Missing frames are NaN with valid=False.

    Speed options (the training features were extracted with the defaults; the fast
    settings were checked against them in scripts/check_fast_mode.py):
      max_side   downscale frames so the long side is at most this many pixels
      det_every  run RTMDet every N frames and track the box from RTMPose keypoints between
      parallel   run MediaPipe and the RTMDet/RTMPose chain in two threads
    """
    from .video import iter_frames, video_info
    info = video_info(video_path)
    fps = info["fps"]
    full_side = max(info["width"], info["height"])
    scale = min(1.0, max_side / full_side) if max_side else 1.0
    need_box = any(b in backends for b in ("rtmpose", "vitpose"))
    det = HandDetector(threads) if need_box else None
    trk = BoxTracker()
    rtm = RTMPoseHand(threads) if "rtmpose" in backends else None
    vit = ViTPoseWholebodyHand(threads) if "vitpose" in backends else None
    mph = MediaPipeHand(fps) if "mediapipe" in backends else None
    pool = None
    if parallel and mph and need_box:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=2)

    ts, boxes, bscores = [], [], []
    out = {b: dict(kpts=[], score=[]) for b in backends}
    if mph:
        out["mediapipe"]["world"] = []
    nan_k, zero_s = np.full((N_KP, 2), np.nan, np.float32), np.zeros(N_KP, np.float32)
    n_total = info["n_frames"]
    prev_rtm = (None, None)

    def box_chain(i, img):
        nonlocal prev_rtm
        box, bs = None, 0.0
        if det:
            tracked = _box_from_kpts(*prev_rtm) if (det_every > 1 and i % det_every) else None
            if tracked is not None:
                box, bs = tracked, -1.0          # -1 marks a tracked (not detected) box
                trk.prev = tracked
            else:
                box, bs = trk.update(*det(img))
        res = {}
        if rtm:
            k, s = rtm(img, box) if box is not None else (nan_k, zero_s)
            res["rtmpose"] = (k, s)
            prev_rtm = (k, s) if box is not None else (None, None)
        if vit:
            res["vitpose"] = vit(img, box) if box is not None else (nan_k, zero_s)
        return box, bs, res

    for i, t, img in iter_frames(video_path, max_seconds=max_seconds,
                                 max_side=max_side if scale < 1 else None):
        ts.append(t)
        if pool:
            fut = pool.submit(box_chain, i, img)
            # MediaPipe picks the hand nearest last frame's box; the current one isn't ready yet
            mp_res = mph(img, t, trk.prev)
            box, bs, res = fut.result()
        else:
            box, bs, res = box_chain(i, img)
            mp_res = mph(img, t, box) if mph else None
        boxes.append(box / scale if box is not None else np.full(4, np.nan, np.float32))
        bscores.append(bs)
        for b, (k, s) in res.items():
            out[b]["kpts"].append(k / scale); out[b]["score"].append(s)
        if mph:
            k, s, w = mp_res
            if k is None:
                k, s, w = nan_k, zero_s, np.full((N_KP, 3), np.nan, np.float32)
            out["mediapipe"]["kpts"].append(k / scale); out["mediapipe"]["score"].append(s)
            out["mediapipe"]["world"].append(w)
        if progress and n_total:
            progress(min(1.0, (i + 1) / n_total))
    if pool:
        pool.shutdown()
    if mph:
        mph.close()
    size = (info["height"], info["width"])
    res = dict(t=np.asarray(ts), fps=fps, size=np.asarray(size),
               box=np.asarray(boxes, np.float32), box_score=np.asarray(bscores, np.float32))
    for b, d in out.items():
        r = {k: np.asarray(v, np.float32) for k, v in d.items()}
        r["valid"] = ~np.isnan(r["kpts"][:, :, 0]).any(1)
        res[b] = r
    return res
