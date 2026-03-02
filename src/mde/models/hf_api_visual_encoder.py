from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover - optional runtime dependency
    InferenceClient = None


class HFAPIVisualEncoder:
    """Hugging Face Inference API wrapper for frame affect + compact embedding."""

    def __init__(
        self,
        affect_model_name: str = "trpakov/vit-face-expression",
        api_token: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> None:
        self.affect_model_name = affect_model_name
        self.allow_fallback = allow_fallback
        self.client = InferenceClient(token=api_token) if InferenceClient is not None else None

    def _fallback(self, frames: list[str]) -> tuple[list[float], list[float], str]:
        lowered = [f.lower() for f in frames]
        downcast = sum(1 for f in lowered if "down" in f or "avoid" in f)
        smile = sum(1 for f in lowered if "smile" in f)
        frozen = sum(1 for f in lowered if "still" in f or "flat" in f)

        total = max(len(frames), 1)
        down_ratio = downcast / total
        smile_ratio = smile / total
        frozen_ratio = frozen / total

        emb = [0.0] * 8
        emb[0] = down_ratio
        emb[1] = smile_ratio
        emb[2] = frozen_ratio

        sad = min(1.0, max(0.0, 0.6 * down_ratio + 0.5 * frozen_ratio - 0.4 * smile_ratio))
        engaged = min(1.0, max(0.0, 0.6 * smile_ratio))
        neutral = min(1.0, max(0.0, 1.0 - sad - engaged))

        summary = "reduced expressivity and downward gaze" if sad > 0.45 else "neutral facial affect"
        return emb, [sad, neutral, engaged], summary

    def _normalize_rows(self, rows: Any) -> list[dict[str, Any]]:
        if hasattr(rows, "tolist"):
            rows = rows.tolist()
        if rows and isinstance(rows[0], list):
            rows = rows[0]
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []

    def _aggregate_scores(self, per_frame: list[list[dict[str, Any]]]) -> dict[str, float]:
        scores: dict[str, float] = {}
        if not per_frame:
            return scores

        for rows in per_frame:
            frame_scores: dict[str, float] = {}
            for row in rows:
                label = str(row.get("label", "")).lower()
                score = float(row.get("score", 0.0))
                frame_scores[label] = frame_scores.get(label, 0.0) + score
            for label, value in frame_scores.items():
                scores[label] = scores.get(label, 0.0) + value

        count = float(len(per_frame))
        return {k: v / count for k, v in scores.items()}

    def _map_affect(self, scores: dict[str, float]) -> list[float]:
        sad = sum(v for k, v in scores.items() if any(x in k for x in ["sad", "fear", "disgust"]))
        neutral = sum(v for k, v in scores.items() if "neutral" in k)
        engaged = sum(v for k, v in scores.items() if any(x in k for x in ["happy", "surprise"]))

        unknown = max(0.0, 1.0 - (sad + neutral + engaged))
        neutral += unknown

        denom = sad + neutral + engaged
        if denom <= 0.0:
            return [0.0, 1.0, 0.0]
        return [sad / denom, neutral / denom, engaged / denom]

    def encode(self, frames: list[str]) -> tuple[list[float], list[float], str]:
        if not frames:
            return [0.0] * 8, [0.0, 1.0, 0.0], "no visual cues"

        if self.client is None:
            return self._fallback(frames)

        per_frame_rows: list[list[dict[str, Any]]] = []
        try:
            for frame in frames:
                path = Path(frame)
                if not path.exists() or not path.is_file():
                    continue
                # This client/provider combination expects a local path/URL or raw bytes.
                # Passing the path string allows the client to infer content type.
                rows = self.client.image_classification(
                    str(path),
                    model=self.affect_model_name,
                )
                parsed = self._normalize_rows(rows)
                if parsed:
                    per_frame_rows.append(parsed)

            if not per_frame_rows:
                return self._fallback(frames)

            scores = self._aggregate_scores(per_frame_rows)
            affect = self._map_affect(scores)

            emb = [0.0] * 8
            for i, key in enumerate(sorted(scores.keys())[:8]):
                emb[i] = scores[key]

            sad, neutral, engaged = affect
            if sad >= max(neutral, engaged):
                summary = "reduced expressivity and downward gaze"
            elif engaged > neutral:
                summary = "higher facial expressivity"
            else:
                summary = "neutral facial affect"

            return emb, affect, summary
        except Exception:
            if not self.allow_fallback:
                raise
            return self._fallback(frames)
