from __future__ import annotations

import tempfile
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime dependency
    cv2 = None


class FacePipeline:
    """Extracts frames, crops dominant face, and filters poor-quality frames."""

    def __init__(
        self,
        fps: float = 1.0,
        max_frames: int = 10,
        min_face_size: int = 64,
        blur_threshold: float = 60.0,
        brightness_min: float = 25.0,
        brightness_max: float = 230.0,
    ) -> None:
        self.fps = fps
        self.max_frames = max_frames
        self.min_face_size = min_face_size
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max

        self._cascade = None
        if cv2 is not None:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if cascade_path.exists():
                self._cascade = cv2.CascadeClassifier(str(cascade_path))

    @property
    def available(self) -> bool:
        return cv2 is not None and self._cascade is not None

    def extract_frames_from_video(self, video_path: str) -> list[str]:
        if not self.available:
            raise RuntimeError("OpenCV face/video pipeline unavailable. Install opencv-python.")

        path = Path(video_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(1, int(round(native_fps / max(self.fps, 0.1))))

        out_dir = Path(tempfile.mkdtemp(prefix="mde_video_frames_"))
        saved: list[str] = []

        idx = 0
        kept = 0
        while cap.isOpened() and kept < self.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_interval == 0:
                frame_path = out_dir / f"frame_{kept:03d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved.append(str(frame_path))
                kept += 1
            idx += 1

        cap.release()
        return saved

    def process_frames(self, frame_paths: list[str]) -> list[str]:
        if not frame_paths:
            return []
        if not self.available:
            return frame_paths[: self.max_frames]

        out_dir = Path(tempfile.mkdtemp(prefix="mde_face_frames_"))
        processed: list[str] = []

        for frame_path in frame_paths:
            if len(processed) >= self.max_frames:
                break

            path = Path(frame_path)
            if not path.exists() or not path.is_file():
                continue

            image = cv2.imread(str(path))
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_face_size, self.min_face_size))

            if len(faces) == 0:
                continue

            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            x0 = max(0, int(x - 0.20 * w))
            y0 = max(0, int(y - 0.25 * h))
            x1 = min(image.shape[1], int(x + 1.20 * w))
            y1 = min(image.shape[0], int(y + 1.15 * h))
            face = image[y0:y1, x0:x1]
            if face.size == 0:
                continue

            blur = cv2.Laplacian(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            brightness = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).mean()
            if blur < self.blur_threshold:
                continue
            if brightness < self.brightness_min or brightness > self.brightness_max:
                continue

            resized = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
            out_path = out_dir / f"face_{len(processed):03d}.jpg"
            cv2.imwrite(str(out_path), resized)
            processed.append(str(out_path))

        if not processed:
            return frame_paths[: self.max_frames]
        return processed
