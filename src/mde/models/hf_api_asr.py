from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover - optional runtime dependency
    InferenceClient = None


class HFAPIAudioTranscriber:
    """Hugging Face Inference API ASR wrapper."""

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        api_token: Optional[str] = None,
        allow_fallback: bool = True,
        provider: str = "hf-inference",
    ) -> None:
        self.model_name = model_name
        self.allow_fallback = allow_fallback
        self.provider = provider
        self.client = None
        if InferenceClient is not None:
            try:
                # Force HF serverless provider to avoid paid-provider auto-routing.
                self.client = InferenceClient(provider=provider, token=api_token)
            except TypeError:
                # Backward compatibility with older huggingface_hub versions.
                self.client = InferenceClient(token=api_token)

    def _normalize_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            text = value.get("text") or value.get("generated_text") or ""
            return str(text).strip()
        return str(value).strip()

    def transcribe(self, audio_path: str) -> str:
        path = Path(audio_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Audio file not found for ASR: {path}")

        if self.client is None:
            if self.allow_fallback:
                return ""
            raise RuntimeError("huggingface_hub is not installed")

        last_error: Exception | None = None

        try:
            if hasattr(self.client, "automatic_speech_recognition"):
                # Prefer local path so the SDK/provider can infer a correct content type.
                out = self.client.automatic_speech_recognition(str(path), model=self.model_name)
                text = self._normalize_text(out)
                if text:
                    return text
        except Exception as exc:  # pragma: no cover - runtime/provider dependent
            last_error = exc
            # Compatibility fallback: some client/provider variants accept raw bytes.
            try:
                with path.open("rb") as f:
                    raw = f.read()
                out = self.client.automatic_speech_recognition(raw, model=self.model_name)
                text = self._normalize_text(out)
                if text:
                    return text
            except Exception as exc2:  # pragma: no cover - runtime/provider dependent
                last_error = exc2

        # Client compatibility fallback for variants exposing speech_to_text.
        try:
            if hasattr(self.client, "speech_to_text"):
                out = self.client.speech_to_text(str(path), model=self.model_name)
                text = self._normalize_text(out)
                if text:
                    return text
        except Exception as exc:  # pragma: no cover - runtime/provider dependent
            last_error = exc
            try:
                with path.open("rb") as f:
                    raw = f.read()
                out = self.client.speech_to_text(raw, model=self.model_name)
                text = self._normalize_text(out)
                if text:
                    return text
            except Exception as exc2:  # pragma: no cover - runtime/provider dependent
                last_error = exc2

        if self.allow_fallback:
            return ""
        if last_error is not None:
            raise RuntimeError(f"ASR transcription failed: {last_error}") from last_error
        raise RuntimeError("No compatible ASR method found on InferenceClient")
