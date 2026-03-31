"""
resist. — Local ONNX Inference
Uses the two models trained in the Colab notebook:
  - band_detector.onnx   : YOLOv8n — finds band bounding boxes
  - color_classifier.onnx: Small CNN — classifies each band's color
"""

from __future__ import annotations
import json, os
import numpy as np
import cv2

# onnxruntime is the only runtime dependency — no PyTorch needed
try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime is not installed. Run: pip install onnxruntime"
    )

# ── Constants ──────────────────────────────────────────────────────────────────
MODELS_DIR   = os.path.join(os.path.dirname(__file__), 'models')
YOLO_PATH    = os.path.join(MODELS_DIR, 'band_detector.onnx')
CNN_PATH     = os.path.join(MODELS_DIR, 'color_classifier.onnx')
CC_JSON      = os.path.join(MODELS_DIR, 'color_classes.json')

YOLO_SIZE    = 640    # input size YOLO was trained at
CNN_SIZE     = 64     # input size CNN was trained at
CONF_THRESH  = 0.35   # minimum YOLO detection confidence
IOU_THRESH   = 0.45   # NMS IoU threshold

# Resistance lookup tables (mirrored from app.py so inference is self-contained)
DIGIT  = dict(black=0,brown=1,red=2,orange=3,yellow=4,
              green=5,blue=6,violet=7,gray=8,grey=8,white=9)
MULT   = dict(black=1,brown=10,red=100,orange=1_000,yellow=10_000,
              green=100_000,blue=1_000_000,violet=10_000_000,
              gray=100_000_000,grey=100_000_000,white=1_000_000_000,
              gold=0.1,silver=0.01)
TOL    = dict(brown='±1%',red='±2%',green='±0.5%',blue='±0.25%',
              violet='±0.1%',gray='±0.05%',grey='±0.05%',
              gold='±5%',silver='±10%',none='±20%')
TEMPCO = dict(black=250,brown=100,red=50,orange=15,yellow=25,
              green=20,blue=10,violet=5,gray=1,grey=1)


# ── ResistorDetector ───────────────────────────────────────────────────────────
class ResistorDetector:
    """
    Loads both ONNX models once and exposes a single .detect(image_bytes) method.
    Thread-safe — create one instance at app startup and reuse it.
    """

    def __init__(self):
        if not os.path.exists(YOLO_PATH):
            raise FileNotFoundError(
                f"Model not found at {YOLO_PATH}\n"
                "Run the Colab notebook first, then copy the .onnx files to inference/models/"
            )
        if not os.path.exists(CNN_PATH):
            raise FileNotFoundError(
                f"Color classifier not found at {CNN_PATH}"
            )

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self._yolo  = ort.InferenceSession(YOLO_PATH,  providers=providers)
        self._cnn   = ort.InferenceSession(CNN_PATH,   providers=providers)

        with open(CC_JSON) as f:
            self._color_classes: list[str] = json.load(f)

        self._yolo_input  = self._yolo.get_inputs()[0].name
        self._cnn_input   = self._cnn.get_inputs()[0].name

        print(f"[detector] YOLO loaded  — {os.path.basename(YOLO_PATH)}")
        print(f"[detector] CNN loaded   — {os.path.basename(CNN_PATH)}")
        print(f"[detector] Colors       — {self._color_classes}")

    # ── Public API ─────────────────────────────────────────────────────────────
    def detect(self, image_bytes: bytes) -> dict:
        """
        Run the full pipeline on raw image bytes.
        Returns a dict matching the same schema as the Gemini API route:
          {bands, ohms, tolerance?, tempco?, band_count, confidence}
        or {error: str} on failure.
        """
        img_bgr = self._decode(image_bytes)
        if img_bgr is None:
            return {'error': 'Could not decode image'}

        boxes, scores = self._run_yolo(img_bgr)
        if len(boxes) == 0:
            return {'error': 'No resistor bands detected in this image'}

        # Sort bands left-to-right by their x-centre
        order   = np.argsort([(b[0]+b[2])/2 for b in boxes])
        bands   = []
        confs   = []
        for i in order:
            crop = self._crop(img_bgr, boxes[i])
            color, conf = self._classify_color(crop)
            bands.append(color)
            confs.append(conf)

        overall_conf = float(np.mean(confs))
        confidence   = 'high' if overall_conf > 0.85 else ('medium' if overall_conf > 0.60 else 'low')

        result = self._calc(bands)
        if 'error' in result:
            return result

        result['confidence'] = confidence
        return result

    # ── Preprocessing ──────────────────────────────────────────────────────────
    @staticmethod
    def _decode(data: bytes):
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    def _preprocess_yolo(self, img: np.ndarray):
        h, w    = img.shape[:2]
        scale   = YOLO_SIZE / max(h, w)
        nh, nw  = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded  = np.full((YOLO_SIZE, YOLO_SIZE, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = resized
        x = padded[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, /255
        x = np.transpose(x, (2, 0, 1))[np.newaxis]         # HWC→NCHW
        return x, scale

    # ── YOLO ───────────────────────────────────────────────────────────────────
    def _run_yolo(self, img: np.ndarray) -> tuple[list, list]:
        inp, scale = self._preprocess_yolo(img)
        raw = self._yolo.run(None, {self._yolo_input: inp})[0]   # (1, 4+nc, 8400)

        # YOLOv8 ONNX output: (batch, 4+num_classes, anchors) — need to transpose
        preds = raw[0].T           # (8400, 4+nc)
        scores = preds[:, 4:].max(axis=1)
        mask   = scores > CONF_THRESH

        if not mask.any():
            return [], []

        boxes_cxcywh = preds[mask, :4]   # (cx, cy, w, h) in YOLO_SIZE coords
        scores_filt  = scores[mask]

        # Convert cx/cy/w/h → x1/y1/x2/y2 in original image coords
        orig_h, orig_w = img.shape[:2]
        boxes_xyxy = np.stack([
            (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2) / scale,
            (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2) / scale,
            (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2) / scale,
            (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2) / scale,
        ], axis=1)

        # Clip to image bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        # NMS
        keep = self._nms(boxes_xyxy, scores_filt, IOU_THRESH)
        return boxes_xyxy[keep].tolist(), scores_filt[keep].tolist()

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas  = (x2 - x1) * (y2 - y1)
        order  = scores.argsort()[::-1]
        keep   = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou < iou_thresh]
        return keep

    # ── Color classifier ───────────────────────────────────────────────────────
    @staticmethod
    def _crop(img: np.ndarray, box: list) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((CNN_SIZE, CNN_SIZE, 3), dtype=np.uint8)
        return crop

    def _classify_color(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        patch = cv2.resize(crop_bgr, (CNN_SIZE, CNN_SIZE))
        # BGR→RGB, normalise to [-1, 1] (matches training transforms)
        x = (patch[:, :, ::-1].astype(np.float32) / 127.5) - 1.0
        x = np.transpose(x, (2, 0, 1))[np.newaxis]

        logits = self._cnn.run(['logits'], {self._cnn_input: x})[0][0]
        # Softmax for confidence
        exp_l  = np.exp(logits - logits.max())
        probs  = exp_l / exp_l.sum()
        idx    = int(np.argmax(probs))
        return self._color_classes[idx], float(probs[idx])

    # ── Resistance calculation ─────────────────────────────────────────────────
    @staticmethod
    def _calc(bands: list[str]) -> dict:
        b = [c.lower().strip() for c in bands]
        n = len(b)
        res: dict = {'bands': b, 'band_count': n}
        try:
            if n == 3:
                res['ohms'] = (DIGIT[b[0]] * 10 + DIGIT[b[1]]) * MULT[b[2]]
            elif n == 4:
                res['ohms'] = (DIGIT[b[0]] * 10 + DIGIT[b[1]]) * MULT[b[2]]
                res['tolerance'] = TOL.get(b[3])
            elif n == 5:
                res['ohms'] = (DIGIT[b[0]] * 100 + DIGIT[b[1]] * 10 + DIGIT[b[2]]) * MULT[b[3]]
                res['tolerance'] = TOL.get(b[4])
            elif n == 6:
                res['ohms'] = (DIGIT[b[0]] * 100 + DIGIT[b[1]] * 10 + DIGIT[b[2]]) * MULT[b[3]]
                res['tolerance'] = TOL.get(b[4])
                tc = TEMPCO.get(b[5])
                if tc:
                    res['tempco'] = f'{tc} ppm/K'
            else:
                return {'error': f'Need 3–6 bands, got {n}'}
        except KeyError as e:
            return {'error': f'Unknown color: {str(e).strip(chr(39))}'}
        return res


# ── Module-level singleton (loaded once at import time) ─────────────────────
_detector: ResistorDetector | None = None

def get_detector() -> ResistorDetector:
    global _detector
    if _detector is None:
        _detector = ResistorDetector()
    return _detector
