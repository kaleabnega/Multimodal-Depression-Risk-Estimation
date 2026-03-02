from __future__ import annotations

import io
import wave
from typing import Any, Optional

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover - optional runtime dependency
    InferenceClient = None


class HFAPIAudioEncoder:
    """Hugging Face Inference API wrapper for audio affect + compact embedding."""

    def __init__(
        self,
        affect_model_name: str = "superb/wav2vec2-base-superb-er",
        sample_rate: int = 16_000,
        api_token: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> None:
        self.affect_model_name = affect_model_name
        self.sample_rate = sample_rate
        self.allow_fallback = allow_fallback
        self.client = InferenceClient(token=api_token) if InferenceClient is not None else None

    def _waveform_to_wav_bytes(self, waveform: list[float]) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)

            data = bytearray()
            for sample in waveform:
                clipped = max(-1.0, min(1.0, float(sample)))
                intval = int(clipped * 32767.0)
                data.extend(intval.to_bytes(2, byteorder="little", signed=True))
            wf.writeframes(bytes(data))
        return buf.getvalue()

    def _fallback(self, waveform: list[float]) -> tuple[list[float], list[float], str]:
        if not waveform:
            return [0.0] * 8, [0.0, 1.0, 0.0], "no audio cues"

        abs_mean = sum(abs(x) for x in waveform) / len(waveform)
        dynamic_range = max(waveform) - min(waveform)
        pause_proxy = sum(1 for x in waveform if abs(x) < 0.02) / len(waveform)

        emb = [0.0] * 8
        emb[0] = abs_mean
        emb[1] = dynamic_range
        emb[2] = pause_proxy

        sad = min(1.0, max(0.0, 0.7 * pause_proxy + 0.5 * (0.15 - abs_mean)))
        activated = min(1.0, max(0.0, 0.8 * dynamic_range - 0.2))
        neutral = min(1.0, max(0.0, 1.0 - sad - activated))

        affect = [sad, neutral, activated]
        summary = "flat tone and slower cadence" if sad >= max(neutral, activated) else "neutral vocal affect"
        return emb, affect, summary

    def _normalize_rows(self, rows: Any) -> list[dict[str, Any]]:
        if hasattr(rows, "tolist"):
            rows = rows.tolist()
        if rows and isinstance(rows[0], list):
            rows = rows[0]
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []

    def _map_affect(self, rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
        label_scores: dict[str, float] = {}
        for row in rows:
            label = str(row.get("label", "")).lower()
            score = float(row.get("score", 0.0))
            label_scores[label] = label_scores.get(label, 0.0) + score

        sad = sum(v for k, v in label_scores.items() if "sad" in k)
        neutral = sum(v for k, v in label_scores.items() if "neutral" in k or "calm" in k)
        activated = sum(v for k, v in label_scores.items() if any(x in k for x in ["ang", "hap", "exc", "surp"]))

        unknown = max(0.0, 1.0 - (sad + neutral + activated))
        neutral += unknown

        denom = sad + neutral + activated
        if denom <= 0.0:
            affect = [0.0, 1.0, 0.0]
        else:
            affect = [sad / denom, neutral / denom, activated / denom]

        emb = [0.0] * 8
        for i, key in enumerate(sorted(label_scores.keys())[:8]):
            emb[i] = label_scores[key]
        return emb, affect

    def encode(self, waveform: list[float]) -> tuple[list[float], list[float], str]:
        if not waveform:
            return [0.0] * 8, [0.0, 1.0, 0.0], "no audio cues"

        if self.client is None:
            return self._fallback(waveform)

        try:
            wav_bytes = self._waveform_to_wav_bytes(waveform)
            rows = self.client.audio_classification(
                wav_bytes,
                model=self.affect_model_name,
            )
            parsed = self._normalize_rows(rows)
            emb, affect = self._map_affect(parsed)
            sad, neutral, activated = affect
            if sad >= max(neutral, activated):
                summary = "flat tone and slower cadence"
            elif activated > neutral:
                summary = "elevated vocal activation"
            else:
                summary = "neutral vocal affect"
            return emb, affect, summary
        except Exception:
            if not self.allow_fallback:
                raise
            return self._fallback(waveform)
