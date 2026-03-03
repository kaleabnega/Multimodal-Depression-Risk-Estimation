# Multimodal Depression Risk Estimation

This repository contains a class-based implementation for a multimodal depression risk estimation system.

## Architecture

- `TextEncoder`: Hugging Face embedding model + optional text risk classifier.
- `AudioEncoder`: Hugging Face speech embedding + emotion classifier.
- `VisualEncoder`: Hugging Face vision embedding + facial expression classifier.
- `MaskedFusionMLP`: fuses modalities with missing-modality masks.
- `SafetyPolicyEngine`: maps score + crisis language to policy state.
- `TemplateResponseGenerator`: returns policy-gated safe responses.
- `DepressionRiskPipeline`: end-to-end orchestrator.

## Run Demo

```bash
PYTHONPATH=src python scripts/run_demo.py \
  --backend hf_api \
  --response-backend guarded_llm \
  --response-model mistralai/Mistral-7B-Instruct-v0.3 \
  --text "I have felt very low this week"
```

For full multimodal pretrained inference:

```bash
PYTHONPATH=src python scripts/run_demo.py \
  --backend hf_api \
  --response-backend guarded_llm \
  --text "I feel empty and tired lately" \
  --audio-wav /absolute/path/to/audio_16khz.wav \
  --frames /absolute/path/to/frame1.jpg /absolute/path/to/frame2.jpg
```

Audio transcription into text branch (ASR):

```bash
PYTHONPATH=src python scripts/run_demo.py \
  --backend hf_api \
  --asr-from-audio \
  --audio-wav /absolute/path/to/audio_16khz.wav
```

Combine typed text + spoken transcript:

```bash
PYTHONPATH=src python scripts/run_demo.py \
  --backend hf_api \
  --text "I want to add context" \
  --asr-from-audio \
  --audio-wav /absolute/path/to/audio_16khz.wav
```

Video-based visual inference with face preprocessing:

```bash
PYTHONPATH=src python scripts/run_demo.py \
  --backend hf_api \
  --text "I feel okay today" \
  --video /absolute/path/to/clip.mp4 \
  --video-fps 1.0 \
  --max-frames 8
```

Notes:
- `--audio-wav` must be a 16kHz WAV file.
- `--asr-from-audio` uses HF ASR API and injects transcript into the text branch.
- `--frames` should be image file paths.
- `--video` extracts frames and runs a local face pipeline (detect/crop/filter) before visual inference.
- Install optional vision dependencies for video/face pipeline: `pip install -e .[vision]`.
- Explicit visual questions (for example, "How is my facial expression?") return a visual-expression summary with confidence when visual cues are available.
- For guarded LLM responses, structured visual context (dominant label, confidence, distribution) is included in the prompt.
- Default backend is `hf_api`. Use `--backend local` for local `transformers` inference.
- Default response backend is `guarded_llm`. Use `--response-backend template` for deterministic templates only.
- Override responder LLM with `--response-model <model_id>` if your provider does not support the default.
- Set token with `--hf-api-token` or environment variable `HF_API_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`).
- `scripts/run_demo.py` auto-loads `.env` from project root, so `HF_API_TOKEN=...` in `.env` works directly.
- Local backend uses `load_pretrained=True` and `allow_fallback=False` by default.
- Add `--allow-fallback` if you want to proceed when model loading fails.
- Add `--local-files-only` to force model loading from local Hugging Face cache only.

## Notes

- Encoders include fallback behavior when model loading fails or dependencies are missing.
- Train/calibrate the fusion head on labeled data (e.g., PHQ-derived labels) before deployment.
- This system is for screening support research, not medical diagnosis.

## Web Interface

A React + Vite web interface is available in `web/`.

Start backend API:

```bash
pip install -e ".[api]"
PYTHONPATH=src python scripts/run_api.py
```

In a second terminal, start frontend:

```bash
cd web
npm install
npm run dev
```

The web UI supports direct video upload for expression questions via `POST /api/chat-upload`.
