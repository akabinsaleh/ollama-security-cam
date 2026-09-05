"""
security_cam.py — Webcam-based motion-detection security system.

Usage:
    python security_cam.py

Dependencies:
    pip install opencv-python

All configuration (email credentials, thresholds, etc.) lives in config.py.
"""

import base64
import collections
import os
import sqlite3
import time
import logging
from datetime import datetime

import cv2
import config

# anthropic is optional — imported lazily inside analyze_appearance().
# Install with:  pip install anthropic

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def open_camera(index: int) -> cv2.VideoCapture:
    """Open the webcam at *index* and return the capture object."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)   # CAP_DSHOW = faster init on Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {index}. "
            "Make sure no other application is using it and the index is correct."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    log.info("Camera opened (index=%d, %dx%d)",
             index,
             int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return cap


def read_frame(cap: cv2.VideoCapture):
    """Read one frame from *cap*, or raise RuntimeError on failure."""
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read frame from camera. Camera may have been disconnected.")
    return frame


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def ensure_captures_dir(path: str) -> None:
    """Create the captures directory if it does not already exist."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot create captures directory '{path}': {exc}") from exc


def save_snapshot(frame, captures_dir: str) -> str:
    """Save *frame* as a JPEG in *captures_dir* with a timestamp filename."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename  = f"person_{timestamp}.jpg"
    filepath  = os.path.join(captures_dir, filename)
    try:
        ok = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise OSError("cv2.imwrite returned False — disk may be full or path invalid.")
    except Exception as exc:
        raise RuntimeError(f"Failed to save snapshot to '{filepath}': {exc}") from exc
    log.info("Snapshot saved → %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------

class VideoRecorder:
    """Records a video clip around each detection event, with pre-roll."""

    def __init__(self, captures_dir: str, fps: float, post_motion_seconds: float):
        self._dir = captures_dir
        self._fps = fps
        self._post_seconds = post_motion_seconds
        self._writer: cv2.VideoWriter | None = None
        self._last_motion_time: float = 0.0
        self._filepath: str = ""

    def start(self, pre_buffer: collections.deque, frame_size: tuple) -> None:
        """Open a new writer and flush the pre-motion buffer into it."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._filepath = os.path.join(self._dir, f"person_{timestamp}.avi")
        w, h = frame_size
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._writer = cv2.VideoWriter(self._filepath, fourcc, self._fps, (w, h))
        for f in pre_buffer:
            self._writer.write(f)
        self._last_motion_time = time.time()
        log.info("Recording started → %s", self._filepath)

    def write(self, frame) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def keep_alive(self) -> None:
        self._last_motion_time = time.time()

    def should_stop(self) -> bool:
        return (time.time() - self._last_motion_time) > self._post_seconds

    def stop(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            log.info("Recording saved → %s", self._filepath)

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def filepath(self) -> str:
        return self._filepath


# ---------------------------------------------------------------------------
# FPS tracker
# ---------------------------------------------------------------------------

class FPSTracker:
    """Rolling-window frames-per-second calculator."""

    def __init__(self, window: int = 30):
        self._times: collections.deque = collections.deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.time())
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ---------------------------------------------------------------------------
# Stage 1 — MOG2 motion pre-filter
# ---------------------------------------------------------------------------

def build_background_subtractor() -> cv2.BackgroundSubtractorMOG2:
    """Return a tuned MOG2 background subtractor used as a fast pre-filter."""
    return cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=config.MOG2_SENSITIVITY,
        detectShadows=True,
    )


# ---------------------------------------------------------------------------
# Stage 2 — Human-scale contour filter
# ---------------------------------------------------------------------------

def detect_human_scale(frame, subtractor: cv2.BackgroundSubtractorMOG2) -> list:
    """Return bounding boxes of human-sized moving blobs.

    Algorithm
    ---------
    1. MOG2 foreground mask  — separates moving pixels from background.
    2. Morphological clean-up — removes noise and small specks.
    3. Blob merge (tall dilation kernel) — joins nearby patches belonging
       to the same person.  Critical for loose garments (thobes, abayas,
       coats) that produce several disconnected foreground regions.
    4. Contour filter — keeps only blobs that satisfy all three of:
         • height  >= MIN_HUMAN_HEIGHT  (tall enough to be a person)
         • area    >= MIN_HUMAN_AREA    (large enough — rules out pets / shadows)
         • w/h     <= MAX_HUMAN_ASPECT  (taller than wide — rules out flat objects)

    Returns a list of (x, y, w, h) tuples ready to draw on screen.
    """
    blurred  = cv2.GaussianBlur(frame, config.BLUR_KERNEL, 0)
    fg_mask  = subtractor.apply(blurred)

    # Binary threshold — drop shadow pixels (127), keep only confident fg (255).
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Small kernel: remove salt-and-pepper noise.
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_small, iterations=2)
    thresh  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k_small, iterations=1)

    # Tall dilation kernel: merge nearby patches into one human-shaped blob.
    # Width 20 px, height 60 px — bridges gaps left by billowing fabric.
    k_merge = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 60))
    thresh  = cv2.dilate(thresh, k_merge, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < config.MIN_HUMAN_AREA:
            continue                          # too small — pet / leaf / shadow

        x, y, w, h = cv2.boundingRect(c)

        if h < config.MIN_HUMAN_HEIGHT:
            continue                          # not tall enough to be a person

        if h > 0 and (w / h) > config.MAX_HUMAN_ASPECT:
            continue                          # too wide — furniture / ground reflection

        boxes.append((x, y, w, h))

    return boxes


# ---------------------------------------------------------------------------
# Annotated display frame
# ---------------------------------------------------------------------------

def draw_overlay(
    frame,
    boxes: list,           # (x, y, w, h) tuples — persons or motion regions
    alert_active: bool,
    fps: float,
    event_count: int,
    cooldown: "AlertCooldown",
    is_recording: bool,
    mode: str,             # "human" or "motion"
    analysis: str = "",    # AI appearance description from Claude
):
    """Return a copy of *frame* with a rich status overlay."""
    annotated = frame.copy()
    h_f, w_f  = annotated.shape[:2]

    # --- Bounding boxes ---
    for (x, y, bw, bh) in boxes:
        # Draw filled semi-transparent rectangle behind label
        label_str = "Person" if mode == "person" else ""
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        if label_str:
            lx, ly = x, max(y - 6, 12)
            cv2.putText(annotated, label_str, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    # --- Status label (top-left) ---
    if alert_active:
        label  = "!! PERSON DETECTED !!" if mode == "person" else "!! MOTION DETECTED !!"
        colour = (0, 0, 255)
    else:
        label  = "Monitoring..."
        colour = (0, 200, 0)
    cv2.putText(annotated, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)

    # --- Detection mode badge ---
    badge = f"Mode: {mode.upper()}"
    cv2.putText(annotated, badge, (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 200, 255), 1, cv2.LINE_AA)

    # --- Event counter ---
    cv2.putText(annotated, f"Events: {event_count}", (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # --- Cooldown countdown ---
    remaining = cooldown.seconds_remaining()
    if remaining > 0:
        cv2.putText(annotated, f"Next alert in: {remaining:.0f}s", (10, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 180, 255), 1, cv2.LINE_AA)

    # --- FPS (top-right) ---
    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(annotated, fps_text, (w_f - tw - 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # --- REC dot (top-right, below FPS) ---
    if is_recording:
        cv2.circle(annotated, (w_f - 20, 50), 8, (0, 0, 255), -1)
        cv2.putText(annotated, "REC", (w_f - 58, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # --- AI appearance analysis panel (bottom strip) ---
    if analysis:
        panel_h = 80
        panel_y = h_f - panel_h
        # Semi-transparent dark background so text is always readable.
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, panel_y), (w_f, h_f), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)
        # Header
        cv2.putText(annotated, "AI Analysis:", (10, panel_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 255), 1, cv2.LINE_AA)
        # Wrap and draw each line of the analysis.
        y_pos = panel_y + 34
        for raw_line in analysis.splitlines():
            raw_line = raw_line.strip()
            if raw_line:
                y_pos = _draw_wrapped(annotated, raw_line, 10, y_pos,
                                      w_f - 20, (230, 230, 230),
                                      scale=0.44, line_h=17)
        # Timestamp sits just above the panel.
        ts_y = panel_y - 6
    else:
        ts_y = h_f - 10

    # --- Timestamp ---
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(annotated, ts, (10, ts_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    return annotated


# ---------------------------------------------------------------------------
# Cooldown tracker
# ---------------------------------------------------------------------------

class AlertCooldown:
    """Prevents alert spam by enforcing a minimum gap between alerts."""

    def __init__(self, cooldown_seconds: float):
        self._cooldown = cooldown_seconds
        self._last_sent: float = 0.0

    def is_ready(self) -> bool:
        return (time.time() - self._last_sent) >= self._cooldown

    def mark_sent(self) -> None:
        self._last_sent = time.time()

    def seconds_remaining(self) -> float:
        return max(0.0, self._cooldown - (time.time() - self._last_sent))


# ---------------------------------------------------------------------------
# AI appearance analysis
# ---------------------------------------------------------------------------

def analyze_appearance(image_path: str) -> str:
    """Send the snapshot to Claude vision and return a description.

    Describes clothing, colours, build, and activity in three short lines.
    Returns an empty string silently if the API key is missing or the call fails.
    """
    if not getattr(config, "ANALYZE_APPEARANCE", False):
        return ""
    api_key = getattr(config, "ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("your-"):
        log.warning("ANTHROPIC_API_KEY not set — skipping appearance analysis.")
        return ""

    try:
        import anthropic as _anthropic

        with open(image_path, "rb") as f:
            img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

        client = _anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # cheapest model — good for testing
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Security camera frame. A person has been detected. "
                            "Describe them in exactly three short lines, no preamble:\n"
                            "Clothing: <colors and garment types>\n"
                            "Appearance: <build, hair, accessories>\n"
                            "Activity: <what they appear to be doing>"
                        ),
                    },
                ],
            }],
        )
        text = next((b.text for b in response.content if b.type == "text"), "").strip()
        log.info("Appearance analysis: %s", text.replace("\n", " | "))
        return text

    except Exception as exc:
        log.warning("Appearance analysis failed: %s", exc)
        return ""


def _draw_wrapped(img, text: str, x: int, y: int, max_w: int,
                  colour, scale: float = 0.45, thickness: int = 1,
                  line_h: int = 18) -> int:
    """Draw word-wrapped text on *img*; return the y position after the last line."""
    line = ""
    for word in text.split():
        test = f"{line} {word}".strip()
        (tw, _), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if tw > max_w and line:
            cv2.putText(img, line, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)
            y += line_h
            line = word
        else:
            line = test
    if line:
        cv2.putText(img, line, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)
        y += line_h
    return y


# ---------------------------------------------------------------------------
# Sound alert
# ---------------------------------------------------------------------------

def play_alert_sound() -> None:
    """Play a short beep. Windows-only; silent no-op elsewhere."""
    try:
        import winsound
        winsound.Beep(1000, 300)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

def log_motion_event(log_file: str, snapshot_path: str | None, video_path: str | None) -> None:
    """Append a timestamped line to the plain-text event log."""
    ts   = datetime.now().isoformat(timespec="seconds")
    line = f"{ts}\tsnapshot={snapshot_path or 'FAILED'}\tvideo={video_path or 'N/A'}\n"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError as exc:
        log.warning("Could not write to event log '%s': %s", log_file, exc)


# ---------------------------------------------------------------------------
# SQLite event database
# ---------------------------------------------------------------------------

DB_FILE = "events.db"


def init_db() -> None:
    """Create the events table if it does not already exist."""
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                camera_index  INTEGER NOT NULL,
                snapshot_path TEXT,
                video_path    TEXT,
                analysis      TEXT,
                box_count     INTEGER DEFAULT 0
            )
        """)


def save_event_to_db(
    timestamp: str,
    camera_index: int,
    snapshot_path: str | None,
    video_path: str | None,
    analysis: str,
    box_count: int,
) -> None:
    """Insert one detection event into the database."""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                "INSERT INTO events "
                "(timestamp, camera_index, snapshot_path, video_path, analysis, box_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (timestamp, camera_index, snapshot_path, video_path, analysis, box_count),
            )
    except sqlite3.Error as exc:
        log.warning("Could not save event to database: %s", exc)


# ---------------------------------------------------------------------------
# Camera open with retry
# ---------------------------------------------------------------------------

def _try_open_camera() -> cv2.VideoCapture | None:
    for attempt in range(1, config.CAMERA_RECONNECT_ATTEMPTS + 1):
        try:
            return open_camera(config.CAMERA_INDEX)
        except RuntimeError as exc:
            log.warning("Camera open attempt %d/%d failed: %s",
                        attempt, config.CAMERA_RECONNECT_ATTEMPTS, exc)
            if attempt < config.CAMERA_RECONNECT_ATTEMPTS:
                time.sleep(config.CAMERA_RECONNECT_DELAY)
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    """Entry point — opens the camera and runs the detection loop."""

    mode = config.DETECTION_MODE   # "human" or "motion"
    log.info("Detection mode: %s", mode)

    init_db()
    ensure_captures_dir(config.CAPTURES_DIR)
    cooldown    = AlertCooldown(config.ALERT_COOLDOWN_SECONDS)
    subtractor  = build_background_subtractor()
    fps_tracker = FPSTracker()
    event_count = 0

    pre_buffer: collections.deque = collections.deque(maxlen=config.VIDEO_PRE_BUFFER_FRAMES)

    recorder = (
        VideoRecorder(
            config.CAPTURES_DIR,
            fps=config.VIDEO_FPS,
            post_motion_seconds=config.VIDEO_POST_MOTION_SECONDS,
        )
        if config.RECORD_VIDEO
        else None
    )

    # --- Open camera ---
    cap = _try_open_camera()
    if cap is None:
        log.critical("Could not open camera after %d attempts. Exiting.",
                     config.CAMERA_RECONNECT_ATTEMPTS)
        return

    # --- Warm up MOG2 ---
    log.info("Warming up background model (2 seconds)…")
    warmup_end = time.time() + 2.0
    while time.time() < warmup_end:
        try:
            frame = read_frame(cap)
        except RuntimeError:
            break
        subtractor.apply(cv2.GaussianBlur(frame, config.BLUR_KERNEL, 0))

    log.info("Monitoring started.  Press Q in the camera window to quit.")

    persist_count   = 0    # consecutive frames with a confirmed detection
    last_analysis   = ""   # most recent AI appearance description

    try:
        while True:
            # --- Read frame (reconnect on failure) ---
            try:
                frame = read_frame(cap)
            except RuntimeError as exc:
                log.error("%s — attempting to reconnect…", exc)
                cap.release()
                cap = _try_open_camera()
                if cap is None:
                    log.critical("Could not reconnect. Giving up.")
                    break
                log.info("Reconnected successfully.")
                continue

            fps_current = fps_tracker.tick()
            pre_buffer.append(frame.copy())

            # ------------------------------------------------------------------
            # Detection
            #
            #  "human"  — MOG2 mask + size/shape filters.  Ignores small
            #              objects; catches any person regardless of clothing.
            #  "motion" — raw MOG2, any movement above MOTION_THRESHOLD.
            # ------------------------------------------------------------------
            if mode == "human":
                boxes = detect_human_scale(frame, subtractor)
            else:   # "motion" — legacy raw detection
                blurred = cv2.GaussianBlur(frame, config.BLUR_KERNEL, 0)
                fg      = subtractor.apply(blurred)
                _, thresh = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
                kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                thresh  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
                thresh  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                big = [c for c in contours if cv2.contourArea(c) >= config.MOTION_THRESHOLD]
                boxes = [cv2.boundingRect(c) for c in big]

            detected_this_frame = len(boxes) > 0

            # Require MOTION_PERSIST_FRAMES consecutive positive frames
            # to reduce single-frame false positives.
            if detected_this_frame:
                persist_count += 1
            else:
                persist_count = 0

            alert_triggered = persist_count >= config.MOTION_PERSIST_FRAMES

            # --- Video recording ---
            if recorder is not None:
                if alert_triggered:
                    if not recorder.is_recording:
                        h_f, w_f = frame.shape[:2]
                        recorder.start(pre_buffer, (w_f, h_f))
                    else:
                        recorder.keep_alive()
                        recorder.write(frame)
                elif recorder.is_recording:
                    recorder.write(frame)
                    if recorder.should_stop():
                        recorder.stop()

            # --- Handle alert ---
            if alert_triggered and cooldown.is_ready():
                event_count += 1
                what = "person(s)" if mode == "person" else "motion"
                log.info(
                    "%s confirmed (%d consecutive frames, %d box(es)) — event #%d",
                    what.capitalize(), persist_count, len(boxes), event_count,
                )

                if config.SOUND_ALERT:
                    play_alert_sound()

                snapshot_path = None
                try:
                    snapshot_path = save_snapshot(frame, config.CAPTURES_DIR)
                except RuntimeError as exc:
                    log.error("Snapshot error: %s", exc)

                if snapshot_path:
                    last_analysis = analyze_appearance(snapshot_path)

                    video_path = recorder.filepath if (recorder and recorder.is_recording) else None

                    save_event_to_db(
                        timestamp=datetime.now().isoformat(timespec="seconds"),
                        camera_index=config.CAMERA_INDEX,
                        snapshot_path=snapshot_path,
                        video_path=video_path,
                        analysis=last_analysis,
                        box_count=len(boxes),
                    )

                    if config.SAVE_EVENT_LOG:
                        log_motion_event(config.EVENT_LOG_FILE, snapshot_path, video_path)

                cooldown.mark_sent()
                persist_count = 0

            elif alert_triggered and not cooldown.is_ready():
                log.debug("Detected but cooldown active (%.0fs remaining)",
                          cooldown.seconds_remaining())

            # --- Display ---
            if config.SHOW_LIVE_FEED:
                display = draw_overlay(
                    frame, boxes, alert_triggered,
                    fps=fps_current,
                    event_count=event_count,
                    cooldown=cooldown,
                    is_recording=recorder.is_recording if recorder else False,
                    mode=mode,
                    analysis=last_analysis,
                )
                cv2.imshow(config.WINDOW_TITLE, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    log.info("Q pressed — shutting down.")
                    break

    except KeyboardInterrupt:
        log.info("Interrupted — shutting down.")

    finally:
        if recorder and recorder.is_recording:
            recorder.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        log.info("Camera released.  Goodbye.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run()
