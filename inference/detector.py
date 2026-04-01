"""
resist. — Local ONNX Inference (single-model)

The jbhepner dataset labels each band with its color as the class name,
so one YOLOv8 model handles both detection AND color identification.
No separate CNN classifier needed.

Files needed in inference/models/:
  band_detector.onnx   — the trained YOLOv8n model
  band_classes.json    — class name list (order must match training)
"""

from __future__ import annotations
import json, os
import numpy as np
import cv2

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("Run: pip install onnxruntime")

MODELS_DIR  = os.path.join(os.path.dirname(__file__), 'models')
YOLO_PATH   = os.path.join(MODELS_DIR, 'band_detector.onnx')
CLASS_PATH  = os.path.join(MODELS_DIR, 'band_classes.json')

YOLO_SIZE   = 640
CONF_THRESH = 0.35
IOU_THRESH  = 0.45

# Classes that are NOT band colors — ignore these detections
NON_COLOR = {'resistor', 'breadboard'}

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


class ResistorDetector:
    """
    Loads the YOLO ONNX model once. Call .detect(image_bytes) per request.
    Thread-safe — create one instance at startup and reuse.
    """

    def __init__(self):
        if not os.path.exists(YOLO_PATH):
            raise FileNotFoundError(
                f"Model not found: {YOLO_PATH}\n"
                "Train in Colab and copy band_detector.onnx to inference/models/"
            )
        if not os.path.exists(CLASS_PATH):
            raise FileNotFoundError(f"Class list not found: {CLASS_PATH}")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self._sess  = ort.InferenceSession(YOLO_PATH, providers=providers)
        self._inp   = self._sess.get_inputs()[0].name

        with open(CLASS_PATH) as f:
            self._classes: list[str] = json.load(f)

        print(f"[detector] Model: {os.path.basename(YOLO_PATH)}")
        print(f"[detector] Classes: {self._classes}")

    def detect(self, image_bytes: bytes) -> dict:
        """
        Full pipeline: bytes → bands → resistance.
        Returns same schema as Gemini route:
          {bands, ohms, tolerance?, tempco?, band_count, confidence}
        or {error: str}
        """
        img = self._decode(image_bytes)
        if img is None:
            return {'error': 'Could not decode image'}

        boxes, scores, class_ids = self._run_yolo(img)

        if not boxes:
            return {'error': 'No resistor bands detected in this image'}

        # Sort bands left-to-right by x-centre
        order = sorted(range(len(boxes)),
                       key=lambda i: (boxes[i][0] + boxes[i][2]) / 2)

        bands = [self._classes[class_ids[i]] for i in order]
        avg_conf = float(np.mean([scores[i] for i in order]))
        confidence = ('high' if avg_conf > 0.75
                      else 'medium' if avg_conf > 0.50 else 'low')

        result = self._calc(bands)
        if 'error' in result:
            return result
        result['confidence'] = confidence
        return result

    # ── private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _decode(data: bytes):
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _preprocess(self, img: np.ndarray):
        h, w = img.shape[:2]
        scale = YOLO_SIZE / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        pad = np.full((YOLO_SIZE, YOLO_SIZE, 3), 114, dtype=np.uint8)
        pad[:nh, :nw] = cv2.resize(img, (nw, nh))
        x = pad[:, :, ::-1].astype(np.float32) / 255.0
        return np.transpose(x, (2, 0, 1))[np.newaxis], scale

    def _run_yolo(self, img: np.ndarray):
        h, w = img.shape[:2]
        inp, scale = self._preprocess(img)
        raw = self._sess.run(None, {self._inp: inp})[0]  # (1, 4+nc, 8400)

        preds   = raw[0].T   # (8400, 4+nc)
        nc      = len(self._classes)
        scores  = preds[:, 4:4+nc].max(axis=1)
        cls_ids = preds[:, 4:4+nc].argmax(axis=1)
        mask    = scores > CONF_THRESH

        if not mask.any():
            return [], [], []

        boxes_cxcywh = preds[mask, :4]
        scores_filt  = scores[mask]
        cls_filt     = cls_ids[mask]

        # Convert to xyxy in original image coordinates
        xyxy = np.stack([
            (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2) / scale,
            (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2) / scale,
            (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2) / scale,
            (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2) / scale,
        ], axis=1)
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)

        # Filter out non-color classes
        color_mask = np.array([
            self._classes[c] not in NON_COLOR for c in cls_filt
        ])
        xyxy        = xyxy[color_mask]
        scores_filt = scores_filt[color_mask]
        cls_filt    = cls_filt[color_mask]

        if not len(xyxy):
            return [], [], []

        keep = self._nms(xyxy, scores_filt)
        return (xyxy[keep].tolist(),
                scores_filt[keep].tolist(),
                cls_filt[keep].tolist())

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas  = (x2 - x1) * (y2 - y1)
        order  = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size:
            i = order[0]; keep.append(i)
            if order.size == 1: break
            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou < IOU_THRESH]
        return keep

    @staticmethod
    def _needs_flip(bands: list[str]) -> bool:
        """Check if bands are reversed (gold/silver in digit position)."""
        if not bands:
            return False
        # Gold/silver should never be first band (always a digit position)
        if bands[0] in ('gold', 'silver'):
            return True
        # For 4+ bands, gold/silver in position 1 also indicates reversal
        if len(bands) >= 4 and bands[1] in ('gold', 'silver'):
            return True
        return False

    @staticmethod
    def _calc(bands: list[str]) -> dict:
        b = [c.lower().strip() for c in bands]
        n = len(b)

        # Auto-flip if gold/silver detected in digit positions
        flipped = False
        if ResistorDetector._needs_flip(b):
            b = b[::-1]
            flipped = True

        res: dict = {'bands': b, 'band_count': n}
        if flipped:
            res['flipped'] = True

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
                if tc: res['tempco'] = f'{tc} ppm/K'
            else:
                return {'error': f'Expected 3–6 bands, got {n}'}
        except KeyError as e:
            color = str(e).strip(chr(39))
            if color in ('gold', 'silver'):
                return {'error': f'Invalid band sequence: {color} cannot be used as a digit band (only as multiplier or tolerance)'}
            return {'error': f'Unknown color: {color}'}
        return res


_detector: ResistorDetector | None = None

def get_detector() -> ResistorDetector:
    global _detector
    if _detector is None:
        _detector = ResistorDetector()
    return _detector
