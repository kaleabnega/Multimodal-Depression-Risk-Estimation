from __future__ import annotations

from mde.utils.math_utils import mean


class AudioEncoder:
    """Placeholder audio encoder for waveform-level summary features."""

    def __init__(self, emb_dim: int = 8) -> None:
        self.emb_dim = emb_dim

    def encode(self, waveform: list[float]) -> tuple[list[float], list[float], str]:
        if not waveform:
            return [0.0] * self.emb_dim, [0.0, 1.0, 0.0], "no audio cues"

        abs_mean = mean([abs(x) for x in waveform])
        dynamic_range = max(waveform) - min(waveform)
        pause_proxy = sum(1 for x in waveform if abs(x) < 0.02) / len(waveform)

        emb = [0.0] * self.emb_dim
        emb[0] = abs_mean
        emb[1] = dynamic_range
        emb[2] = pause_proxy

        # [sad, neutral, activated]
        sad = min(1.0, max(0.0, 0.7 * pause_proxy + 0.5 * (0.15 - abs_mean)))
        activated = min(1.0, max(0.0, 0.8 * dynamic_range - 0.2))
        neutral = min(1.0, max(0.0, 1.0 - sad - activated))

        summary = "flat tone and slower cadence" if sad > 0.45 else "neutral vocal affect"
        return emb, [sad, neutral, activated], summary
