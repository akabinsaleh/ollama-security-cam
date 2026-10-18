"""Command-line interface.

    python -m facecam enroll --name Me     # capture your face (look at the camera)
    python -m facecam run                  # live detection + recognition
    python -m facecam image photo.jpg      # check a single image
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .config import DB_PATH, ENROLL_SAMPLES, MIN_ENROLL_CONFIDENCE
from .models import detect, embed, identify, load_db, load_models, save_db

def draw(frame, face, label, color):
    x, y, w, h = face[:4].astype(int)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    for i in range(5):  # eyes, nose, mouth corners
        cv2.circle(frame, tuple(face[4 + 2 * i:6 + 2 * i].astype(int)), 2, color, -1)
    cv2.putText(frame, label, (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {index}")
    return cap


def cmd_enroll(args):
    detector, recognizer = load_models()
    cap = open_camera(args.camera)
    samples, last = [], 0.0
    print("Look at the camera; slowly turn your head a little. Press q to abort.")
    while len(samples) < ENROLL_SAMPLES:
        ok, frame = cap.read()
        if not ok:
            break
        faces = detect(detector, frame)
        # Only enroll when exactly one clear face is visible.
        if len(faces) == 1 and faces[0][-1] > MIN_ENROLL_CONFIDENCE and time.time() - last > 0.4:
            samples.append(embed(recognizer, frame, faces[0])[0])
            last = time.time()
        for f in faces:
            draw(frame, f, f"enrolling {len(samples)}/{ENROLL_SAMPLES}", (0, 200, 255))
        cv2.imshow("enroll", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if len(samples) < ENROLL_SAMPLES:
        sys.exit("Enrollment incomplete; nothing saved.")
    db = load_db()
    db[args.name] = np.array(samples)
    save_db(db)
    print(f"Saved {len(samples)} samples for '{args.name}' to {DB_PATH}")


def cmd_run(args):
    detector, recognizer = load_models()
    db = load_db()
    if not db:
        sys.exit("No enrolled faces. Run: python -m facecam enroll --name Me")
    cap = open_camera(args.camera)
    print("Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for f in detect(detector, frame):
            name, score = identify(recognizer, embed(recognizer, frame, f), db)
            if name:
                draw(frame, f, f"{name} ({score:.2f})", (0, 220, 0))
            else:
                draw(frame, f, f"Unknown ({score:.2f})", (0, 0, 230))
        cv2.imshow("face detect", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def cmd_image(args):
    detector, recognizer = load_models()
    db = load_db()
    frame = cv2.imread(args.path)
    if frame is None:
        sys.exit(f"Cannot read {args.path}")
    faces = detect(detector, frame)
    print(f"{len(faces)} face(s) found")
    for f in faces:
        if db:
            name, score = identify(recognizer, embed(recognizer, frame, f), db)
            print(f"  {name or 'Unknown'} (score {score:.2f})")
            draw(frame, f, name or "Unknown", (0, 220, 0) if name else (0, 0, 230))
        else:
            draw(frame, f, "face", (0, 220, 0))
    out = Path(args.path).with_suffix(".out.jpg")
    cv2.imwrite(str(out), frame)
    print(f"Annotated image saved to {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll")
    e.add_argument("--name", default="Me")
    e.add_argument("--camera", type=int, default=0)
    e.set_defaults(fn=cmd_enroll)
    r = sub.add_parser("run")
    r.add_argument("--camera", type=int, default=0)
    r.set_defaults(fn=cmd_run)
    i = sub.add_parser("image")
    i.add_argument("path")
    i.set_defaults(fn=cmd_image)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
