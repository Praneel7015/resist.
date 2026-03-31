# resist.

> Identify any resistor instantly — upload a photo or pick bands manually.

A clean, minimal web app that decodes resistor color bands using AI vision. Built with Flask and a zero-dependency frontend.

![resist UI](https://img.shields.io/badge/stack-Flask%20%2B%20Gemini-orange?style=flat-square) ![license](https://img.shields.io/badge/license-MIT-blue?style=flat-square)

---

## Features

- **AI photo detection** — point your camera or drop an image; Gemini Vision reads the bands in any lighting
- **Manual band picker** — live resistor SVG updates as you pick colors; supports 3, 4, 5, and 6-band resistors
- **Correct resistance math** — separate formulas for each band count, proper tolerance and tempco display
- **Formatted output** — Ω / kΩ / MΩ / GΩ with tolerance and confidence indicator
- **Responsive** — works on mobile and desktop, rear camera supported

---

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python 3.11 · Flask 3 · Pillow |
| AI detection | Google Gemini 1.5 Flash (free tier) |
| Frontend | Vanilla JS · CSS custom properties · no frameworks |
| Server | Waitress (production) · Flask dev server (local) |
| Container | Docker |

---

## Getting started

### 1. Get a free Gemini API key

Go to [aistudio.google.com](https://aistudio.google.com) → **Get API key** → Create a key in a new project. The free tier allows 15 requests/minute and 1 million tokens/day — more than enough for personal use.

### 2. Clone and set up

```bash
git clone https://github.com/your-username/resist
cd resist

python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure your key

```bash
cp .env.example .env
# Edit .env and set your key:
# GEMINI_API_KEY=AIza...
```

### 4. Run

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Usage

**Photo mode** — click "Browse Files" or "Use Camera", select a resistor image, press **Analyze Bands**. The app sends the image to Gemini Vision which identifies the band colors, then calculates resistance client-side.

**Manual mode** — switch to the Manual tab, select how many bands your resistor has (3–6), then click each band's color. The resistor diagram and resistance value update in real time.

### Tips for better photo detection

- Good, even lighting matters most — avoid shadows across the bands
- Photograph the resistor on a plain white or dark background
- Get as close as possible so the body fills the frame
- If confidence shows as "low", try manual mode instead

---

## Project structure

```
resist/
├── app.py                 # Flask server: /, /health, /analyze
├── requirements.txt       # Python dependencies (no heavy ML libs)
├── Dockerfile             # Container image
├── .env.example           # API key template
├── templates/
│   └── index.html         # Single-page UI
└── static/
    ├── style.css          # Dark theme, CSS variables, responsive
    └── app.js             # Upload, camera, manual picker, state machine
```

---

## Roadmap

The current version uses the Gemini API for detection. The long-term goal is a fully self-hosted model with no API dependency. Here's the path:

### Phase 0 — current
Gemini 1.5 Flash via API. Free tier. Works on any photo.

### Phase 1 — data collection
Gather a resistor dataset. The [Roboflow Universe](https://universe.roboflow.com/search?q=resistor) has several annotated resistor datasets you can export in YOLO format. Supplement with your own photos.

### Phase 2 — custom model
Train [YOLOv8n](https://docs.ultralytics.com/models/yolov8/) on Google Colab (free GPU) to detect the band bounding boxes. Then train a small CNN color classifier on the cropped band images. Export both to ONNX for fast CPU inference.

Colab training notebook: [Ultralytics YOLOv8 tutorial](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)

Roboflow resistor dataset: [universe.roboflow.com — resistor](https://universe.roboflow.com/search?q=resistor+color+band)

### Phase 3 — self-hosted
Replace the Gemini API call with local ONNX inference. Zero API costs, runs offline, works in airgapped environments.


---

## Deployment

### Local development

```bash
python app.py
# Runs on http://127.0.0.1:5000
```

### Docker

```bash
docker build -t resist .
docker run -p 8080:8080 -e GEMINI_API_KEY=AIza... resist
```

### AWS App Runner 

App Runner scales to near-zero when idle (0.25 vCPU / 512 MB floor) so it won't burn credits sitting unused.

1. Push this repo to GitHub
2. Open the [AWS Console → App Runner](https://console.aws.amazon.com/apprunner)
3. **Create service** → Source: **Repository** → connect your GitHub repo and branch
4. Build: use **Dockerfile** (auto-detected)
5. Port: `8080`
6. Environment variable: `GEMINI_API_KEY` = your key
7. Health check path: `/health`
8. Deploy — App Runner builds and redeploys on every push

Estimated cost with AWS credits: near zero for low-traffic personal use.


---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google AI Studio API key |
| `PORT` | No | Server port (default: 8080 in Docker, 5000 local) |

---

## Switching AI providers

The detection logic is isolated in the `/analyze` route in `app.py`. To swap providers, only this function needs changing. The rest of the app (calculation logic, UI, manual mode) is unaffected.

The prompt used for detection:
```
Identify the resistor color bands in this image, reading left-to-right
starting from the end nearest a lead/leg. Return ONLY valid JSON:
{"bands":["color1","color2",...],"confidence":"high|medium|low"}
```

---

## Contributing

Pull requests welcome. If you've trained a custom YOLO model for band detection and want to contribute it, please open an issue first to discuss the integration approach.

---

## License

MIT
