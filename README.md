# Security Camera — Motion Detection & Alert System

A production-quality Python application that turns any webcam into a real-time
motion-detection security system with email alerting.

---

## Features

| Feature | Detail |
|---|---|
| Motion detection | MOG2 background subtraction + contour filtering |
| False-positive suppression | Configurable area threshold + multi-frame persistence |
| Snapshot capture | JPEG saved to `captures/` with timestamp filename |
| Email alert | Snapshot attached and sent via Gmail SMTP (TLS) |
| Cooldown system | Minimum gap between alerts (default 45 s) |
| Live feed | Annotated window with bounding boxes and status label |
| Graceful exit | Press **Q** in the window or **Ctrl+C** in the terminal |

---

## Quick start

### 1. Install Python 3.9+

Download from https://python.org if you don't have it.

### 2. Install dependencies

```bash
pip install -r requirements.txt
# or directly:
pip install opencv-python
```

### 3. Configure

Open **`config.py`** and fill in:

```python
SENDER_EMAIL   = "you@gmail.com"
RECEIVER_EMAIL = "you@gmail.com"   # can be the same address
APP_PASSWORD   = "xxxx xxxx xxxx xxxx"   # Gmail App Password — NOT your login password
```

Everything else (thresholds, cooldown, camera index) is already set to
sensible defaults but is easy to tune.

#### Getting a Gmail App Password

1. Go to <https://myaccount.google.com/security>
2. Enable **2-Step Verification** (required).
3. Search for **"App passwords"** in the same page.
4. Select *Mail* + *Other (custom name)* → **Generate**.
5. Copy the 16-character password into `config.py`.

> The App Password grants access to Gmail sending only — it is completely
> independent of your account password and can be revoked at any time.

### 4. Run

```bash
python security_cam.py
```

A camera window will open.  Walk in front of the webcam — you should see red
bounding boxes appear and a snapshot saved to the `captures/` folder.  An email
is sent on the first trigger (and then at most once per cooldown window).

Press **Q** in the camera window, or **Ctrl+C** in the terminal, to stop.

---

## Configuration reference (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam index (0 = built-in, 1+ = external) |
| `MOTION_THRESHOLD` | `1500` | Min contour area (px²) to count as motion |
| `MOG2_SENSITIVITY` | `25` | Background-subtractor var-threshold (lower = more sensitive) |
| `BLUR_KERNEL` | `(21, 21)` | Gaussian blur kernel — reduces noise |
| `MOTION_PERSIST_FRAMES` | `3` | Consecutive frames needed before alert fires |
| `ALERT_COOLDOWN_SECONDS` | `45` | Minimum gap between two email alerts |
| `CAPTURES_DIR` | `"captures"` | Folder where snapshots are saved |
| `SHOW_LIVE_FEED` | `True` | Show annotated camera window |

---

## Project structure

```
seacurty cam/
├── security_cam.py   # Main application — detection loop, camera I/O, email
├── config.py         # All tunable settings and credentials
├── requirements.txt  # pip dependencies
├── README.md         # This file
└── captures/         # Created automatically; snapshot JPEGs stored here
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Cannot open camera" | Check no other app (Zoom, Teams) is using the webcam; try `CAMERA_INDEX = 1` |
| Too many false positives | Raise `MOTION_THRESHOLD` or `MOTION_PERSIST_FRAMES` |
| Misses real motion | Lower `MOTION_THRESHOLD` or `MOG2_SENSITIVITY` |
| "SMTP authentication failed" | Use an App Password, not your Gmail login password |
| Email never arrives | Check spam folder; verify `SENDER_EMAIL` / `RECEIVER_EMAIL` |
| Window flickers/lags | Lower capture resolution in `open_camera()` or set `SHOW_LIVE_FEED = False` |

---

## Future improvements

### Higher-priority ideas

- **Multi-zone detection** — define rectangular regions of interest (ROI) so you
  only monitor a doorway, not the whole frame.
- **SMS alerts** — integrate [Twilio](https://twilio.com) (`twilio` package) for
  an SMS alongside the email.
- **Push notifications** — use the [Pushover](https://pushover.net) or
  [ntfy.sh](https://ntfy.sh) API for instant mobile notifications.

### Storage

- **Cloud storage** — upload snapshots to AWS S3 (`boto3`) or Google Drive
  (`google-api-python-client`) so footage survives local disk failure.
- **Video clip instead of still** — buffer the 5 seconds surrounding a trigger
  and save as MP4 using `cv2.VideoWriter`.

### Intelligence

- **Person detection** — add a YOLO or MobileNet SSD model (via `ultralytics`
  or `cv2.dnn`) so alerts only fire when a *person* is detected, not a cat.
- **Face recognition** — use `face_recognition` to skip alerts for known
  household members.

### Monitoring & UI

- **Web dashboard** — serve a live MJPEG stream with Flask/FastAPI so you can
  watch the feed from a phone browser on the same network.
- **Alert log** — write trigger events to a SQLite database for history review.
- **Prometheus metrics** — export motion-count and alert-count metrics for
  Grafana dashboards.

### Deployment

- **Run as a background service** — use `systemd` (Linux) or Task Scheduler
  (Windows) so the system starts automatically on boot without a desktop session.
- **Headless mode** — `SHOW_LIVE_FEED = False` already supports this; combine
  with the web dashboard for remote viewing.
- **Docker container** — containerise for easy deployment on a Raspberry Pi or
  home server (use `opencv-python-headless` in the image).
