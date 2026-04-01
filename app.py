"""
resist. — Flask App
Detection priority:
  1. Local ONNX models (inference/models/*.onnx)  — zero API cost
  2. Gemini 2.5 Flash fallback                    — if models not trained yet
"""
import os, io, base64, json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
ALLOWED = {'.jpg', '.jpeg', '.png', '.webp'}

# ── Try to load the local ONNX detector ────────────────────────────────────────
_local_detector = None
_local_error    = None

try:
    from inference.detector import get_detector
    _local_detector = get_detector()
    print("[app] Local ONNX models loaded — running fully offline")
except FileNotFoundError as e:
    _local_error = str(e)
    print(f"[app] Local models not found — will use Gemini API")
    print(f"[app]    {e}")
except Exception as e:
    _local_error = str(e)
    print(f"[app] Could not load local models: {e} — will use Gemini API")

# ── Helpers ────────────────────────────────────────────────────────────────────
def _to_jpeg_b64(raw: bytes) -> str:
    img = Image.open(io.BytesIO(raw))
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGB')
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    if max(img.width, img.height) > 1400:
        img.thumbnail((1400, 1400), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format='JPEG', quality=88)
    return base64.standard_b64encode(out.getvalue()).decode()


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

GEMINI_PROMPT = (
    "You are an expert electronics technician. Identify the resistor color bands "
    "in this image, reading left-to-right starting from the end nearest a lead/leg. "
    "The tolerance band (gold or silver) is typically on the right side. "
    "Return ONLY a valid JSON object — no markdown, no explanation:\n"
    '{"bands":["color1","color2",...],"confidence":"high|medium|low"}\n'
    "Valid colors: black brown red orange yellow green blue violet gray white gold silver\n"
    'If no resistor is visible: {"error":"brief reason"}'
)


def _needs_flip(bands: list) -> bool:
    """Check if bands are reversed (gold/silver in digit position)."""
    if not bands:
        return False
    if bands[0] in ('gold', 'silver'):
        return True
    if len(bands) >= 4 and bands[1] in ('gold', 'silver'):
        return True
    return False


def _calc(bands: list) -> dict:
    b = [c.lower().strip() for c in bands]
    n = len(b)

    # Auto-flip if gold/silver detected in digit positions
    flipped = False
    if _needs_flip(b):
        b = b[::-1]
        flipped = True

    res = {'bands': b, 'band_count': n}
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
            return {'error': f'Need 3–6 bands, got {n}'}
    except KeyError as e:
        color = str(e).strip(chr(39))
        if color in ('gold', 'silver'):
            return {'error': f'Invalid band sequence: {color} cannot be used as a digit band (only as multiplier or tolerance)'}
        return {'error': f'Unknown color: {color}'}
    return res


def _gemini_detect(image_bytes: bytes) -> dict:
    """Fallback to Gemini 2.5 Flash when local models are not available."""
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        return {'error': 'No local models found and GEMINI_API_KEY is not set. '
                         'Train the models first or add a Gemini API key.'}
    import urllib.request, urllib.error
    b64 = _to_jpeg_b64(image_bytes)
    url  = (f'https://generativelanguage.googleapis.com/v1beta/'
            f'models/gemini-2.5-flash:generateContent?key={key}')
    payload = json.dumps({
        'contents': [{'parts': [
            {'inline_data': {'mime_type': 'image/jpeg', 'data': b64}},
            {'text': GEMINI_PROMPT}
        ]}],
        'generationConfig': {
            'temperature': 0,
            'maxOutputTokens': 1024,
            'response_mime_type': 'application/json',
        }
    }).encode()
    try:
        req = urllib.request.Request(url, data=payload,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        # Check for safety blocks or empty candidates
        if not data.get('candidates'):
            reason = data.get('promptFeedback', {}).get('blockReason', 'unknown')
            return {'error': f'Gemini blocked the request: {reason}'}

        candidate = data['candidates'][0]
        finish = candidate.get('finishReason', '')
        if finish not in ('STOP', 'MAX_TOKENS', ''):
            return {'error': f'Gemini did not finish normally (reason: {finish})'}

        raw = candidate['content']['parts'][0]['text'].strip()
        # Strip markdown fences if present (some model versions still add them)
        raw = raw.strip('`')
        if raw.lower().startswith('json'):
            raw = raw[4:].strip()

        detection = json.loads(raw)

    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        return {'error': f'Gemini API error {e.code}: {body}'}
    except json.JSONDecodeError as e:
        return {'error': f'Could not parse Gemini response as JSON: {e}'}
    except Exception as e:
        return {'error': f'Detection failed: {e}'}

    if 'error' in detection:
        return detection

    bands = detection.get('bands', [])
    if not bands:
        return {'error': 'No bands found in image'}

    result = _calc(bands)
    if 'error' not in result:
        result['confidence'] = detection.get('confidence', 'medium')
        result['source'] = 'gemini'
    return result


# ── Routes ──────────────────────────────────────────────────────────────────────
@app.get('/health')
def health():
    return jsonify({
        'status': 'ok',
        'mode': 'local' if _local_detector else 'gemini'
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.post('/analyze')
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    f   = request.files['image']
    ext = os.path.splitext(secure_filename(f.filename or 'x.jpg'))[1].lower()
    if ext not in ALLOWED:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    image_bytes = f.read()
    use_gemini = request.form.get('use_gemini', '').lower() == 'true'

    # ── Path A: local ONNX models (unless Gemini override) ─────────────────────
    if _local_detector is not None and not use_gemini:
        result = _local_detector.detect(image_bytes)
        if 'error' in result:
            return jsonify(result), 422
        result['source'] = 'local'
        return jsonify(result)

    # ── Path B: Gemini API fallback ────────────────────────────────────────────
    result = _gemini_detect(image_bytes)
    if 'error' in result:
        code = 422 if 'No' in result['error'] else 500
        return jsonify(result), code
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# AWS Lambda handler
from mangum import Mangum
from asgiref.wsgi import WsgiToAsgi

asgi_app = WsgiToAsgi(app)
handler = Mangum(asgi_app)