import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "opencv_zoo" / "models"

# 1. Main connection and assets paths configurations
# Camera stream, e.g. rtsp://<user>:<password>@<host>:554/<path>; set RTSP_URL in the environment
RTSP_URL = os.environ.get("RTSP_URL", "")
DETECTOR_WEIGHTS = MODELS_DIR / "face_detection_yunet" / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_WEIGHTS = MODELS_DIR / "face_recognition_sface" / "face_recognition_sface_2021dec.onnx"
# Known people, either as faces/<person>.jpg or as a folder faces/<person>/ holding several photos
FACES_DIR = BASE_DIR / "faces"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# SFace cosine similarity: >= 0.363 is the published same-identity threshold
COSINE_THRESHOLD = 0.363
RECONNECT_DELAY_S = 2
# Stop shrinking a reference photo once its longest side is this small
MIN_REFERENCE_SIDE = 320
# How often the faces folder is rescanned for added, changed or removed photos
RESCAN_INTERVAL_S = 2


def check_weights(path):
    """Exit with a useful message if a weights file is missing or is still a Git LFS pointer."""
    if not path.is_file():
        sys.exit(f"Error: weights file not found: {path}")
    with open(path, "rb") as f:
        if f.read(40).startswith(b"version https://git-lfs"):
            sys.exit(
                f"Error: {path} is a Git LFS pointer, not the real model.\n"
                "Run 'git lfs install && git lfs pull' inside opencv_zoo to fetch the weights."
            )


def largest_face(faces):
    # Each row is [x, y, w, h, 10 landmark coords, score]
    return max(faces, key=lambda f: f[2] * f[3])


def detect(detector, image):
    detector.setInputSize((image.shape[1], image.shape[0]))
    _, faces = detector.detect(image)
    return faces


def reference_photos(faces_dir):
    """Yield (person name, photo path) for loose photos and for photos inside per-person folders."""
    for entry in sorted(faces_dir.iterdir()):
        if entry.is_dir():
            # faces/<person>/<anything>.jpg: every photo in the folder belongs to that person
            for photo in sorted(entry.iterdir()):
                if photo.is_file() and photo.suffix.lower() in IMAGE_EXTENSIONS:
                    yield entry.name, photo
        elif entry.suffix.lower() in IMAGE_EXTENSIONS:
            # faces/<person>.jpg
            yield entry.stem, entry


def embed_photo(path, detector, recognizer):
    """Return the L2-normalised embedding of the largest face in the photo, or None if no face is found."""
    img = cv2.imread(str(path))
    faces = detect(detector, img) if img is not None else None
    # YuNet misses faces that fill most of a high-resolution photo, so retry on smaller copies
    while faces is None and img is not None and max(img.shape[:2]) > MIN_REFERENCE_SIDE:
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        faces = detect(detector, img)
    if faces is None:
        return None
    embedding = recognizer.feature(recognizer.alignCrop(img, largest_face(faces))).flatten()
    return embedding / np.linalg.norm(embedding)


class KnownFaces:
    """Known-face embeddings that follow the faces folder while the pipeline is running."""

    def __init__(self, faces_dir, detector, recognizer):
        self.faces_dir = faces_dir
        self.detector = detector
        self.recognizer = recognizer
        self.names = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self._cache = {}  # photo path -> ((mtime, size), embedding or None)
        self._loaded = []  # (photo path, (mtime, size)) of the photos behind the current embeddings
        self._next_scan = 0.0

    def refresh(self):
        """Rescan the folder, embedding only new or modified photos. Return True if the known faces changed."""
        cache, names, embeddings, loaded = {}, [], [], []
        try:
            for name, path in reference_photos(self.faces_dir):
                stat = path.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
                cached = self._cache.get(path)
                if cached is None or cached[0] != signature:
                    cached = (signature, embed_photo(path, self.detector, self.recognizer))
                    if cached[1] is None:
                        print(f"Warning: Skipping {path.relative_to(self.faces_dir)}, no readable face found.")
                cache[path] = cached
                if cached[1] is not None:
                    names.append(name)
                    embeddings.append(cached[1])
                    loaded.append((path, signature))
        except OSError:
            # Folder or photo changed mid-scan; keep the current faces and try again next time
            return False
        self._cache = cache
        if loaded == self._loaded:
            return False
        self.names, self.embeddings, self._loaded = names, np.array(embeddings), loaded
        return True

    def refresh_if_due(self):
        if time.monotonic() < self._next_scan:
            return False
        changed = self.refresh()
        self._next_scan = time.monotonic() + RESCAN_INTERVAL_S
        return changed

    def describe(self):
        people = sorted(set(self.names))
        if not people:
            return f"No known faces yet; add photos to '{self.faces_dir}' and they are picked up automatically."
        return f"Loaded {len(self.names)} photo(s) of {len(people)} known face(s): {', '.join(people)}"

    def match(self, embedding):
        """Return (name or None, best cosine score) for a face embedding."""
        if not self.names:
            return None, 0.0
        cosine_scores = self.embeddings @ (embedding / np.linalg.norm(embedding))
        best = int(np.argmax(cosine_scores))
        score = float(cosine_scores[best])
        return (self.names[best] if score >= COSINE_THRESHOLD else None), score


def main():
    parser = argparse.ArgumentParser(description="YuNet + SFace live face recognition POC")
    parser.add_argument("--source", default=RTSP_URL, help="RTSP URL, video file, or webcam index")
    parser.add_argument("--faces", default=str(FACES_DIR), help="folder of known faces: <person>.jpg files and/or <person>/ folders of photos")
    parser.add_argument("--no-display", action="store_true", help="print matches instead of opening a window")
    args = parser.parse_args()
    if not args.source:
        print("Error: No video source. Set the RTSP_URL environment variable or pass --source.")
        return 1

    check_weights(DETECTOR_WEIGHTS)
    check_weights(RECOGNIZER_WEIGHTS)

    # Initialize the YuNet detector deep learning module
    detector = cv2.FaceDetectorYN.create(str(DETECTOR_WEIGHTS), "", (320, 320), 0.9, 0.3, 5000)

    # Initialize the SFace recognition extractor module
    recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_WEIGHTS), "")

    # Load and map the known identity reference images
    faces_dir = Path(args.faces)
    try:
        faces_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error: Cannot use known faces folder '{faces_dir}': {e}")
        return 1

    known = KnownFaces(faces_dir, detector, recognizer)
    known.refresh_if_due()
    print(known.describe())

    # Initialize NVR live stream matrix capture
    source = int(args.source) if args.source.isdigit() else args.source
    is_live = isinstance(source, int) or "://" in source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open the video source.")
        return 1

    print("Facial Recognition Pipeline Active. Press 'q' on the viewport window to quit.")

    try:
        while True:
            # Pick up photos added to, changed in or removed from the faces folder
            if known.refresh_if_due():
                print(known.describe())

            ret, frame = cap.read()
            if not ret:
                if not is_live:
                    break
                print("Warning: Dropped live network stream sync frame, reconnecting...")
                cap.release()
                time.sleep(RECONNECT_DELAY_S)
                cap = cv2.VideoCapture(source)
                continue

            # Compute face geometries on live capture
            live_faces = detect(detector, frame)

            if live_faces is not None:
                for face in live_faces:
                    # Isolate box coordinate layouts
                    x, y, w, h = face[0:4].astype(np.int32)

                    # Align tracking target array matrix
                    aligned_live_face = recognizer.alignCrop(frame, face)
                    live_embedding = recognizer.feature(aligned_live_face).flatten()

                    # Cross-reference cosine calculation matrices against every known face
                    name, cosine_score = known.match(live_embedding)

                    if name is not None:
                        label = f"{name} ({cosine_score:.2f})"
                        color = (0, 255, 0)  # Green border box
                    else:
                        label = f"UNKNOWN ({cosine_score:.2f})"
                        color = (0, 0, 255)  # Red border box

                    if args.no_display:
                        print(f"{label} at x={x} y={y} w={w} h={h}")
                        continue

                    # Render graphics layout boundaries onto screen
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if not args.no_display:
                cv2.imshow("Live Biometric Stream Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
