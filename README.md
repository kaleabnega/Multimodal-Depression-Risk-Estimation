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
PYTHONPATH=src python scripts/run_demo.py
```

The demo currently runs with `load_pretrained=False` for deterministic offline execution.
Set `load_pretrained=True` in `scripts/run_demo.py` to use real Hugging Face models.

## Notes

- Encoders include fallback behavior when model loading fails or dependencies are missing.
- Train/calibrate the fusion head on labeled data (e.g., PHQ-derived labels) before deployment.
- This system is for screening support research, not medical diagnosis.
