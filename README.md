# Multimodal Depression Risk Estimation (Modular Skeleton)

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

Notes:
- `--audio-wav` must be a 16kHz WAV file.
- `--frames` should be image file paths.
- Default backend is `hf_api`. Use `--backend local` for local `transformers` inference.
- Default response backend is `guarded_llm`. Use `--response-backend template` for deterministic templates only.
- Set token with `--hf-api-token` or environment variable `HF_API_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`).
- `scripts/run_demo.py` auto-loads `.env` from project root, so `HF_API_TOKEN=...` in `.env` works directly.
- Local backend uses `load_pretrained=True` and `allow_fallback=False` by default.
- Add `--allow-fallback` if you want to proceed when model loading fails.
- Add `--local-files-only` to force model loading from local Hugging Face cache only.

## Notes

- Encoders include fallback behavior when model loading fails or dependencies are missing.
- Train/calibrate the fusion head on labeled data (e.g., PHQ-derived labels) before deployment.
- This system is for screening support research, not medical diagnosis.
