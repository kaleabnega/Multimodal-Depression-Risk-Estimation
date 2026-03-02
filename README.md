# Multimodal Depression Risk Estimation (Modular Skeleton)

This repository contains a class-based baseline implementation for a multimodal depression risk estimation system.

## Architecture

- `TextEncoder`: normalizes text and produces embedding + text risk prior.
- `AudioEncoder`: derives simple paralinguistic proxy features.
- `VisualEncoder`: derives simple frame-level affect proxy features.
- `MaskedFusionMLP`: fuses modalities with missing-modality masks.
- `SafetyPolicyEngine`: maps score + crisis language to policy state.
- `TemplateResponseGenerator`: returns policy-gated safe responses.
- `DepressionRiskPipeline`: end-to-end orchestrator.

## Run Demo

```bash
PYTHONPATH=src python scripts/run_demo.py
```

## Notes

- Current encoders/fusion are placeholders designed for clean modular iteration.
- Replace placeholder encoders with pretrained models and train fusion on labeled data.
- This system is for screening support research, not medical diagnosis.
