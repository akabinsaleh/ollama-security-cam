from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "faces.npz"

MODEL_URLS = {
    "yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

MATCH_THRESHOLD = 0.363  # SFace cosine threshold recommended by OpenCV
ENROLL_SAMPLES = 15
MIN_ENROLL_CONFIDENCE = 0.9
