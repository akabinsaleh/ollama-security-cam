"""Model download/loading and the detect / embed / identify primitives."""
import urllib.request

import cv2
import numpy as np

from .config import DATA_DIR, DB_PATH, MATCH_THRESHOLD, MODEL_URLS, MODELS_DIR


def fetch_models():
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in MODEL_URLS.items():
        path = MODELS_DIR / name
        if not path.exists() or path.stat().st_size < 10_000:
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
    return str(MODELS_DIR / "yunet.onnx"), str(MODELS_DIR / "sface.onnx")


def load_models():
    det_path, rec_path = fetch_models()
    detector = cv2.FaceDetectorYN.create(det_path, "", (320, 320), 0.8, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(rec_path, "")
    return detector, recognizer


def detect(detector, frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    return [] if faces is None else faces


def embed(recognizer, frame, face):
    return recognizer.feature(recognizer.alignCrop(frame, face))


def load_db():
    if not DB_PATH.exists():
        return {}
    data = np.load(DB_PATH, allow_pickle=False)
    return {k: data[k] for k in data.files}


def save_db(db):
    DATA_DIR.mkdir(exist_ok=True)
    np.savez(DB_PATH, **db)


def identify(recognizer, feat, db):
    """Return (name, best_score). name is None if nobody matches."""
    best_name, best = None, -1.0
    for name, samples in db.items():
        for ref in samples:
            s = recognizer.match(feat, ref.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE)
            if s > best:
                best_name, best = name, s
    return (best_name if best >= MATCH_THRESHOLD else None), best
