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

PROMPT = (
    "You are an expert electronics technician. Identify the resistor color bands "
    "in this image, reading left-to-right starting from the end nearest a lead/leg. "
    "The tolerance band (gold or silver) is typically on the right side. "
    "Return ONLY a valid JSON object — no markdown, no explanation:\n"
    '{"bands":["color1","color2",...],"confidence":"high|medium|low"}\n'
    "Valid colors (lowercase only): black brown red orange yellow green blue violet gray white gold silver\n"
    'If no resistor is visible or bands cannot be determined: {"error":"brief reason"}'
)

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

def _calc(bands: list) -> dict:
    b = [c.lower().strip() for c in bands]
    n = len(b)
    res = {'bands': b, 'band_count': n}
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

@app.get('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/')
def index():
    return render_template('index.html')

@app.post('/analyze')
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    f = request.files['image']
    ext = os.path.splitext(secure_filename(f.filename or 'x.jpg'))[1].lower()
    if ext not in ALLOWED:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        return jsonify({'error': 'GEMINI_API_KEY is not configured on this server'}), 500

    try:
        b64 = _to_jpeg_b64(f.read())
    except Exception as e:
        return jsonify({'error': f'Cannot read image: {e}'}), 400

    try:
        import urllib.request, urllib.error
        url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}'
        payload = json.dumps({
            'contents': [{
                'parts': [
                    {'inline_data': {'mime_type': 'image/jpeg', 'data': b64}},
                    {'text': PROMPT}
                ]
            }],
            'generationConfig': {'temperature': 0, 'maxOutputTokens': 256}
        }).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())

        raw = data['candidates'][0]['content']['parts'][0]['text'].strip()
        raw = raw.strip('`').strip()
        if raw.startswith('json'):
            raw = raw[4:].strip()
        detection = json.loads(raw)

    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return jsonify({'error': f'Gemini API error {e.code}: {body[:200]}'}), 500
    except json.JSONDecodeError:
        return jsonify({'error': 'Model returned unexpected response format'}), 500
    except Exception as e:
        return jsonify({'error': f'Detection failed: {e}'}), 500

    if 'error' in detection:
        return jsonify({'error': detection['error']}), 422

    bands = detection.get('bands', [])
    if not bands:
        return jsonify({'error': 'No bands found in image'}), 422

    result = _calc(bands)
    if 'error' in result:
        return jsonify(result), 422

    result['confidence'] = detection.get('confidence', 'medium')
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
