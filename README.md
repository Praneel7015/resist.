# resist.

> Identify any resistor instantly — upload a photo, use your camera, or pick bands manually.

A clean, minimal web app that decodes resistor color bands. Runs fully offline using trained ONNX models, with an optional Gemini API fallback.

[![GitHub stars](https://img.shields.io/github/stars/praneel7015/resist?style=flat-square)](https://github.com/praneel7015/resist)
![stack](https://img.shields.io/badge/stack-Flask%20%2B%20YOLOv8%20%2B%20ONNX-orange?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-blue?style=flat-square)

---

## How it works

```
Photo → YOLOv8n (detects band bounding boxes) → resistance formula → result
```

Model trained on Google Colab (free T4 GPU), exported to ONNX. Zero API costs. Runs on CPU.

---

## Features

- **AI photo detection** — works in any lighting once models are trained
- **Live camera** — capture and analyze in one tap
- **Manual band picker** — live resistor SVG, 3/4/5/6-band support, updates in real time
- **Auto band orientation** — automatically flips reversed bands (gold/silver detected first)
- **Correct resistance math** — separate formulas per band count, tolerance and tempco display
- **Offline-first** — ONNX inference, no internet needed after setup
- **Gemini override** — toggle to use Gemini Vision AI instead of local models

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/praneel7015/resist
cd resist

python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2a. Run with Gemini (no training required)

Get a free key at [aistudio.google.com](https://aistudio.google.com):

```bash
cp .env.example .env
# Edit .env: GEMINI_API_KEY=AIza...
python app.py
```

### 2b. Run with local ONNX models (after training)

```bash
# Place trained models in:
# inference/models/band_detector.onnx
# inference/models/band_classes.json

python app.py   # auto-detects models and runs offline
```

You can also check the "Use Gemini AI" toggle in the UI to override local detection and use the Gemini API instead.

---

## Training your own models

### Overview

The detection pipeline uses **YOLOv8n** to find where each band is on the resistor body (object detection).

### Step 1 — Get the Colab notebook

Open [`notebooks/resist_train.ipynb`](notebooks/resist_train.ipynb) in Google Colab.

Or open directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praneel7015/resist/blob/main/notebooks/resist_train.ipynb)

### Step 2 — Set runtime to T4 GPU

In Colab: **Runtime → Change runtime type → T4 GPU**

### Step 3 — Get a Roboflow API key

1. Create a free account at [roboflow.com](https://roboflow.com)
2. Profile icon → **Settings → Roboflow API** → copy your key
3. Paste it into the notebook cell

Dataset used: [Resistor Band Detection](https://universe.roboflow.com/jbhepner/resistor-and-band-detection) — annotated resistor images with per-band bounding boxes and color labels.

### Step 4 — Run all cells

The notebook (~25 minutes on T4) will:
1. Download and inspect the dataset
2. Train YOLOv8n for color detection
3. Export model to ONNX
4. Save files to your Google Drive

### Step 5 — Download and deploy

Copy from `Google Drive → resist_models/` into your project:

```
resist/
└── inference/
    └── models/
        ├── band_detector.onnx       (~11 MB)
        └── band_classes.json
```

Restart Flask — it will auto-detect the models and switch to offline mode.

---

## Project structure

```
resist/
├── app.py                    # Flask server — auto-selects local vs Gemini
├── requirements.txt
├── Dockerfile
├── .env.example
├── notebooks/
│   └── resist_train.ipynb    # Full Colab training notebook
├── inference/
│   ├── detector.py           # ONNX inference — YOLOv8 pipeline
│   └── models/               # Put your .onnx files here (not in git)
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── app.js
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Only without local models | Free key from aistudio.google.com |
| `PORT` | No | Server port (default 8080 in Docker, 5000 local) |

---

## Deployment

### Docker

```bash
docker build -t resist .
docker run -p 8080:8080 resist
# Add -e GEMINI_API_KEY=... if using Gemini fallback
```

### AWS App Runner

App Runner scales near-zero when idle.

1. Push to GitHub
2. [AWS Console → App Runner](https://console.aws.amazon.com/apprunner) → Create service
3. Source: **Repository** → connect GitHub → select this repo
4. Build: **Dockerfile** (auto-detected)
5. Port: `8080`
6. Environment variable: `GEMINI_API_KEY` (only needed without local models)
7. Health check: `/health`
8. Deploy — auto-redeploys on every push to main

### AWS Lambda + API Gateway (true serverless)

Add `mangum` to requirements.txt, then:

```python
# bottom of app.py
from mangum import Mangum
handler = Mangum(app)
```

## Roadmap

- [x] Manual band picker (3–6 bands, live SVG preview)
- [x] Camera capture mode
- [x] Gemini Vision API integration
- [x] YOLOv8 pipeline (Colab notebook)
- [x] ONNX export for zero-dependency inference
- [x] Auto band orientation (flips reversed gold/silver)
- [x] Gemini override toggle in UI
- [ ] Ohm's law calculator
- [ ] LED resistor calculator (supply voltage + LED specs → resistor value)
- [ ] Voltage divider calculator
- [ ] Series / parallel resistance calculator
- [ ] E12 / E24 / E96 nearest standard value finder
- [ ] SMD resistor code decoder
- [ ] Scan history

---

## Tips for better detection

- Good even lighting — avoid shadows across the bands
- Plain background (white or black) behind the resistor
- Get close so the resistor body fills most of the frame
- If confidence shows "low", try the Manual tab or toggle on Gemini AI

---

## Contributing

Pull requests welcome. If you've improved the training pipeline or have a better dataset, please open an issue first.

---

## License

MIT — see [LICENSE](LICENSE)
