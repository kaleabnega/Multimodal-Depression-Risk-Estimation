from __future__ import annotations

import argparse
import os
import wave
from pathlib import Path

from mde.core.types import UserInput
from mde.models.audio_encoder import AudioEncoder
from mde.models.fusion import MaskedFusionMLP
from mde.models.hf_api_audio_encoder import HFAPIAudioEncoder
from mde.models.hf_api_text_encoder import HFAPITextEncoder
from mde.models.hf_api_visual_encoder import HFAPIVisualEncoder
from mde.models.multimodal_encoder import MultimodalEncoder
from mde.models.text_encoder import TextEncoder
from mde.models.visual_encoder import VisualEncoder
from mde.services.pipeline import DepressionRiskPipeline
from mde.services.policy import SafetyPolicyEngine
from mde.services.response import GuardedLLMResponseGenerator, TemplateResponseGenerator


def _load_dotenv(dotenv_path: str = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _read_wav_16khz_mono(path: str) -> list[float]:
    wav_path = Path(path)
    if not wav_path.exists() or not wav_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        num_frames = wf.getnframes()
        raw = wf.readframes(num_frames)

    if sample_rate != 16_000:
        raise ValueError(
            f"Expected 16kHz WAV input, got {sample_rate}Hz at {wav_path}. "
            "Resample to 16000 Hz before inference."
        )

    if sample_width not in (1, 2, 4):
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    samples: list[float] = []
    bytes_per_frame = sample_width * num_channels
    max_int = float((1 << (8 * sample_width - 1)) - 1)

    for i in range(0, len(raw), bytes_per_frame):
        frame_bytes = raw[i : i + bytes_per_frame]
        channel_values: list[float] = []
        for ch in range(num_channels):
            start = ch * sample_width
            chunk = frame_bytes[start : start + sample_width]

            if sample_width == 1:
                value = (chunk[0] - 128) / 128.0
            else:
                intval = int.from_bytes(chunk, byteorder="little", signed=True)
                value = intval / max_int if max_int else 0.0

            channel_values.append(max(-1.0, min(1.0, value)))

        mono = sum(channel_values) / max(len(channel_values), 1)
        samples.append(mono)

    return samples


def build_pipeline(
    *,
    backend: str = "hf_api",
    response_backend: str = "guarded_llm",
    load_pretrained: bool = True,
    local_files_only: bool = False,
    allow_fallback: bool = False,
    api_token: str | None = None,
) -> DepressionRiskPipeline:
    if backend == "hf_api":
        encoder = MultimodalEncoder(
            text_encoder=HFAPITextEncoder(
                api_token=api_token,
                allow_fallback=allow_fallback,
            ),
            audio_encoder=HFAPIAudioEncoder(
                api_token=api_token,
                allow_fallback=allow_fallback,
            ),
            visual_encoder=HFAPIVisualEncoder(
                api_token=api_token,
                allow_fallback=allow_fallback,
            ),
        )
        fusion = MaskedFusionMLP(text_dim=384, audio_dim=8, visual_dim=8)
    else:
        encoder = MultimodalEncoder(
            text_encoder=TextEncoder(
                load_pretrained=load_pretrained,
                local_files_only=local_files_only,
                allow_fallback=allow_fallback,
            ),
            audio_encoder=AudioEncoder(
                load_pretrained=load_pretrained,
                local_files_only=local_files_only,
                allow_fallback=allow_fallback,
            ),
            visual_encoder=VisualEncoder(
                load_pretrained=load_pretrained,
                local_files_only=local_files_only,
                allow_fallback=allow_fallback,
            ),
        )
        fusion = MaskedFusionMLP(text_dim=384, audio_dim=768, visual_dim=768)

    if response_backend == "guarded_llm":
        responder = GuardedLLMResponseGenerator(
            api_token=api_token,
            allow_fallback=allow_fallback,
            use_llm_for_high_risk=False,
        )
    else:
        responder = TemplateResponseGenerator()

    return DepressionRiskPipeline(
        encoder=encoder,
        fusion=fusion,
        policy=SafetyPolicyEngine(low=0.30, high=0.70),
        responder=responder,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multimodal depression risk inference")
    parser.add_argument("--text", required=True, help="User text input")
    parser.add_argument(
        "--backend",
        default="hf_api",
        choices=["hf_api", "local"],
        help="Encoder backend: Hugging Face Inference API or local transformers",
    )
    parser.add_argument(
        "--response-backend",
        default="guarded_llm",
        choices=["guarded_llm", "template"],
        help="Response generator backend",
    )
    parser.add_argument("--audio-wav", help="Path to 16kHz WAV audio file")
    parser.add_argument("--frames", nargs="*", default=None, help="Frame image file paths")
    parser.add_argument(
        "--hf-api-token",
        default=None,
        help="Hugging Face API token (or set HF_API_TOKEN/HUGGINGFACE_HUB_TOKEN)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load Hugging Face models only from local cache",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow deterministic fallback encoders if pretrained loading fails",
    )
    return parser.parse_args()


def main() -> None:
    _load_dotenv()
    args = _parse_args()
    api_token = args.hf_api_token or os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    pipeline = build_pipeline(
        backend=args.backend,
        response_backend=args.response_backend,
        load_pretrained=True,
        local_files_only=args.local_files_only,
        allow_fallback=args.allow_fallback,
        api_token=api_token,
    )

    audio = _read_wav_16khz_mono(args.audio_wav) if args.audio_wav else None
    frames: list[str] | None = None
    if args.frames:
        validated: list[str] = []
        for frame in args.frames:
            frame_path = Path(frame)
            if not frame_path.exists() or not frame_path.is_file():
                raise FileNotFoundError(f"Frame file not found: {frame_path}")
            validated.append(str(frame_path))
        frames = validated

    user_input = UserInput(
        text=args.text,
        audio=audio,
        frames=frames,
    )

    output = pipeline.run_user_input(user_input)

    print(f"risk_score={output.fusion.risk_score:.3f}")
    print(f"policy_state={output.policy.state.value}")
    print(f"reasons={output.policy.reasons}")
    print(f"modality_mask={output.fusion.modality_mask}")
    print(f"response={output.response}")


if __name__ == "__main__":
    main()
