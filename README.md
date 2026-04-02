# Multimodal Depression Risk Estimation

A multimodal mental-health support prototype that combines text, audio, and visual cues to:

- estimate a risk score,
- apply a safety policy,
- generate guarded, empathetic responses.

This project is built for **screening-support and conversational assistance research**, not medical diagnosis.

## What This Project Does

The system supports different input combinations:

- text only,
- text + audio,
- text + video,
- text + audio + video,
- audio-to-text via ASR.

At runtime, it:

1. encodes each available modality,
2. fuses modality signals into a risk score,
3. maps risk + language checks to a policy state,
4. generates a response (guarded LLM or template fallback).

For uploaded audio, the system can:

- run ASR (Whisper via HF Inference API) to recover spoken text,
- inject transcript into the text branch for response generation,
- optionally use acoustic features when uploaded audio is 16kHz WAV.

## Core Capabilities

- Hugging Face API-backed text/audio/visual encoders.
- Optional local transformer-based encoder path.
- Video frame extraction + face preprocessing pipeline.
- ASR module to inject transcript into text branch.
- Visual-expression-aware response mode.
- Crisis-aware policy routing.
- Web API + React UI.
- UI upload support for video and audio attachments.
- Response telemetry (`response_source`, `response_model`) to verify LLM vs fallback path.

## Repository Structure

- `src/mde/models/`: modality encoders and fusion.
- `src/mde/services/`: policy, response generation, pipeline orchestration.
- `src/mde/utils/`: vision pipeline utilities.
- `src/mde/api/server.py`: FastAPI backend.
- `src/mde/runtime.py`: shared pipeline builder.
- `scripts/run_demo.py`: CLI runner.
- `scripts/run_api.py`: API runner.
- `web/`: React + Vite frontend.
- `tests/`: basic pipeline/response tests.

## Architecture Summary

### 1) Input Layer

`UserInput` supports:

- `text: str` (required after ASR merge step),
- `audio: list[float] | None`,
- `frames: list[str] | None`.

### 2) Modality Encoders

- Text encoder -> `text_embedding`, `text_risk`.
- Audio encoder -> `audio_embedding`, `audio_affect_probs`.
- Visual encoder -> `visual_embedding`, `visual_affect_probs`.

### 3) Fusion

`MaskedFusionMLP` combines embeddings and modality mask into `risk_score`.

Note: fusion is currently placeholder-weighted (not yet trained on labeled data).

### 4) Policy

Risk score + crisis language checks -> policy states:

- `NORMAL_SUPPORT`
- `GENTLE_MONITORING`
- `HIGH_RISK_SUPPORT`
- `CRISIS_PROTOCOL`

### 5) Response

- `GuardedLLMResponseGenerator` (primary)
- `TemplateResponseGenerator` (fallback)

Response metadata is exposed to API/UI:

- `response_source` (e.g. `llm`, `template_fallback_error`)
- `response_model`

## Local Setup

### Python backend

```bash
cd /path/to/multimodal-depression-risk-estimation
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -e ".[api]"
pip install -e ".[vision]"   # optional but needed for video face pipeline
```

### Environment variables

Create `.env` in repo root:

```env
HF_API_TOKEN=hf_xxx
# optional
MDE_RESPONSE_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

## Run Backend API

```bash
python scripts/run_api.py
```

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

You should see `hf_token_loaded` and active `response_model`.

## Run Frontend

```bash
cd web
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## API Endpoints

### `POST /api/chat` (JSON)

For text/path-based payloads.

Example:

```bash
curl -s http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"text":"How are you?","debug":true}'
```

### `POST /api/chat-upload` (multipart)

For browser uploads (video/audio).

Supported form fields:

- `text` (optional when ASR is enabled and transcript is produced)
- `video_file` (optional)
- `audio_file` (optional)
- `asr_from_audio` (`true/false`, default `true`)
- `asr_model` (default `openai/whisper-large-v3`)
- `debug` (`true/false`)
- `video_fps`, `max_frames`, `skip_face_pipeline`

Example:

```bash
curl -X POST "http://127.0.0.1:8000/api/chat-upload" \
  -F "text=How is my facial expression?" \
  -F "video_file=@/absolute/path/to/test-video.mp4"
```

Audio ASR example:

```bash
curl -X POST "http://127.0.0.1:8000/api/chat-upload" \
  -F "audio_file=@/absolute/path/to/sample.m4a" \
  -F "asr_from_audio=true" \
  -F "debug=true"
```

## CLI Usage

```bash
PYTHONPATH=src python scripts/run_demo.py --help
```

Common examples:

Text only:

```bash
PYTHONPATH=src python scripts/run_demo.py \
  --backend hf_api \
  --text "I feel a bit overwhelmed"
```

Video expression question:

```bash
PYTHONPATH=src python scripts/run_demo.py \
  --backend hf_api \
  --text "How is my facial expression?" \
  --video /absolute/path/to/clip.mp4 \
  --video-fps 2 \
  --max-frames 8 \
  --debug
```

ASR from audio:

```bash
PYTHONPATH=src python scripts/run_demo.py \
  --backend hf_api \
  --asr-from-audio \
  --audio-wav /absolute/path/to/audio_16khz.wav
```

Notes:

- CLI `--audio-wav` path is for 16kHz WAV acoustic branch.
- API/UI upload can accept common compressed formats for ASR (mp3/m4a/ogg/flac/wav).

## Deployment

### Backend on Railway

Recommended:

- Build: `pip install -r requirements.txt`
- Start: `uvicorn mde.api.server:app --host 0.0.0.0 --port $PORT --app-dir src`

Make sure Railway environment includes:

- `HF_API_TOKEN`
- optionally `MDE_RESPONSE_MODEL`

### Frontend on Netlify

For this repo layout:

- Base directory: `web`
- Build command: `npm run build`
- Publish directory: `dist`

Set frontend env var:

- `VITE_API_BASE_URL=https://<your-railway-domain>`

Also ensure backend CORS allowlist includes your Netlify domain (no trailing slash).

## ASR Provider Notes

- ASR is configured through Hugging Face `InferenceClient`.
- The ASR wrapper is set to use `provider="hf-inference"` to avoid paid-provider auto-routing (for example `fal-ai` 402 errors).
- ASR model/provider availability still depends on your HF account/token and enabled providers.

## Current Limitations

- Fusion network is placeholder-weighted (not trained/calibrated yet).
- Risk score is useful for flow testing, not clinical inference.
- Visual quality and provider/model availability can affect expression output.
- If ASR fails or returns empty transcript, audio content will not contribute lexical meaning unless text is also provided.
