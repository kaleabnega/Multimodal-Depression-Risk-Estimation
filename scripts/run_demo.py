from __future__ import annotations

import argparse
import os
import wave
from pathlib import Path

from mde.core.types import UserInput
from mde.models.hf_api_asr import HFAPIAudioTranscriber
from mde.runtime import build_pipeline
from mde.utils.vision_pipeline import FacePipeline


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multimodal depression risk inference")
    parser.add_argument("--text", default="", help="User text input")
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
    parser.add_argument(
        "--response-model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="LLM model id for guarded_llm response backend",
    )
    parser.add_argument("--audio-wav", help="Path to 16kHz WAV audio file")
    parser.add_argument(
        "--asr-from-audio",
        action="store_true",
        help="Transcribe --audio-wav with HF ASR and inject transcript into text branch",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3",
        help="ASR model id used when --asr-from-audio is set",
    )
    parser.add_argument("--video", help="Path to video file for visual cue extraction")
    parser.add_argument(
        "--video-fps",
        type=float,
        default=1.0,
        help="Frame extraction rate from video input",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Maximum number of frames used for visual inference",
    )
    parser.add_argument(
        "--skip-face-pipeline",
        action="store_true",
        help="Skip local face detection/cropping and use raw frames",
    )
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print intermediate multimodal features for troubleshooting",
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
        response_model=args.response_model,
    )

    audio = _read_wav_16khz_mono(args.audio_wav) if args.audio_wav else None
    final_text = args.text.strip()
    if args.asr_from_audio:
        if not args.audio_wav:
            raise ValueError("--asr-from-audio requires --audio-wav")
        transcriber = HFAPIAudioTranscriber(
            model_name=args.asr_model,
            api_token=api_token,
            allow_fallback=args.allow_fallback,
        )
        transcript = transcriber.transcribe(args.audio_wav).strip()
        if transcript:
            final_text = f"{final_text}\n{transcript}".strip() if final_text else transcript

    if not final_text:
        raise ValueError("Text input is required. Provide --text, or use --asr-from-audio with valid speech.")

    frames: list[str] | None = None
    face_pipeline = FacePipeline(fps=args.video_fps, max_frames=args.max_frames)

    video_frames: list[str] = []
    if args.video:
        video_frames = face_pipeline.extract_frames_from_video(args.video)

    if video_frames:
        frames = video_frames

    if args.frames:
        validated: list[str] = []
        for frame in args.frames:
            frame_path = Path(frame)
            if not frame_path.exists() or not frame_path.is_file():
                raise FileNotFoundError(f"Frame file not found: {frame_path}")
            validated.append(str(frame_path))
        frames = (frames or []) + validated

    if frames and not args.skip_face_pipeline:
        frames = face_pipeline.process_frames(frames)
    elif frames:
        frames = frames[: args.max_frames]

    user_input = UserInput(
        text=final_text,
        audio=audio,
        frames=frames,
    )

    output = pipeline.run_user_input(user_input)

    print(f"risk_score={output.fusion.risk_score:.3f}")
    print(f"policy_state={output.policy.state.value}")
    print(f"reasons={output.policy.reasons}")
    print(f"modality_mask={output.fusion.modality_mask}")
    print(f"response={output.response}")
    if args.debug:
        print("--- debug ---")
        print(f"text_len={len(final_text)}")
        print(f"audio_samples={len(audio) if audio is not None else 0}")
        print(f"num_frames_inferred={len(frames) if frames is not None else 0}")
        print(f"text_risk={output.features.text_risk}")
        print(f"audio_affect_probs={output.features.audio_affect_probs}")
        print(f"visual_affect_probs={output.features.visual_affect_probs}")
        print(f"visual_embedding_head={output.features.visual_embedding[:5] if output.features.visual_embedding else None}")


if __name__ == "__main__":
    main()
