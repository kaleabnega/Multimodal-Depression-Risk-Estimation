from __future__ import annotations


class VisualEncoder:
    """Placeholder visual encoder using frame metadata tokens."""

    def __init__(self, emb_dim: int = 8) -> None:
        self.emb_dim = emb_dim

    def encode(self, frames: list[str]) -> tuple[list[float], list[float], str]:
        if not frames:
            return [0.0] * self.emb_dim, [0.0, 1.0, 0.0], "no visual cues"

        lowered = [f.lower() for f in frames]
        downcast = sum(1 for f in lowered if "down" in f or "avoid" in f)
        smile = sum(1 for f in lowered if "smile" in f)
        frozen = sum(1 for f in lowered if "still" in f or "flat" in f)

        total = max(len(frames), 1)
        down_ratio = downcast / total
        smile_ratio = smile / total
        frozen_ratio = frozen / total

        emb = [0.0] * self.emb_dim
        emb[0] = down_ratio
        emb[1] = smile_ratio
        emb[2] = frozen_ratio

        sad = min(1.0, max(0.0, 0.6 * down_ratio + 0.5 * frozen_ratio - 0.4 * smile_ratio))
        engaged = min(1.0, max(0.0, 0.6 * smile_ratio))
        neutral = min(1.0, max(0.0, 1.0 - sad - engaged))

        summary = "reduced expressivity and downward gaze" if sad > 0.45 else "neutral facial affect"
        return emb, [sad, neutral, engaged], summary
