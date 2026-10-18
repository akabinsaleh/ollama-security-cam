"""
security_cam.py — Webcam-based motion-detection security system.

Usage:
    python security_cam.py

Dependencies:
    pip install -r requirements.txt
    A running Ollama install with a vision model (see OLLAMA_MODEL in config.py).

All configuration (thresholds, Ollama model, notifications) lives in config.py.
"""

import base64
import collections
import json
import os
import queue
import sqlite3
import threading
import time
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime

import cv2
import config

# winotify (Windows toast notifications) is optional and imported lazily
# inside send_notification().

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
        label_str = "Person" if mode == "human" else ""
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        if label_str:
            lx, ly = x, max(y - 6, 12)
            cv2.putText(annotated, label_str, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    # --- Status label (top-left) ---
    if alert_active:
        label  = "!! PERSON DETECTED !!" if mode == "human" else "!! MOTION DETECTED !!"
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
        panel_h = 100   # fits the verdict line plus three description lines
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
# Local AI analysis (Ollama)
# ---------------------------------------------------------------------------

# The model is forced to reply with exactly this JSON shape, so we never have
# to parse free text.
OLLAMA_SCHEMA = {
    "type": "object",
    "properties": {
        "person_present": {"type": "boolean"},
        "confidence":     {"type": "integer"},
        "clothing":       {"type": "string"},
        "appearance":     {"type": "string"},
        "activity":       {"type": "string"},
    },
    "required": ["person_present", "confidence", "clothing", "appearance", "activity"],
}

OLLAMA_PROMPT = (
    "This is a frame from a security camera. Decide whether a real, live person "
    "is visible. Posters, photos, mannequins, reflections and shadows are not people.\n"
    "Reply with JSON only:\n"
    "person_present: true or false.\n"
    "confidence: integer 0 to 100. How sure you are that a real person is visible "
    "(0 = certainly nobody, 100 = certainly a person).\n"
    "clothing, appearance, activity: short phrases about the person. "
    "Use an empty string if nobody is visible."
)


@dataclass
class AIResult:
    """Outcome of one analysis. status is 'done', 'failed' or 'skipped'."""
    status: str
    person_present: bool | None = None
    confidence: int | None = None
    text: str = ""          # human-readable summary shown on screen and in the admin app
    raw: str = ""           # exact model reply, kept for debugging
    latency_ms: int | None = None
    model: str = ""
    error: str = ""


def _encode_snapshot(image_path: str) -> str:
    """Return the snapshot as a base64 JPEG, shrunk to OLLAMA_IMAGE_WIDTH."""
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read snapshot '{image_path}'")
    h, w = img.shape[:2]
    target_w = int(config.OLLAMA_IMAGE_WIDTH)
    if w > target_w:
        img = cv2.resize(img, (target_w, max(1, int(h * target_w / w))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Could not encode snapshot as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _format_analysis(person: bool, confidence: int, data: dict) -> str:
    lines = [f"Person: {'yes' if person else 'no'} ({confidence}/100)"]
    for key in ("clothing", "appearance", "activity"):
        value = str(data.get(key) or "").strip()
        if value:
            lines.append(f"{key.capitalize()}: {value}")
    return "\n".join(lines)


def analyze_snapshot(image_path: str) -> AIResult:
    """Ask the local Ollama vision model about the snapshot.

    Never raises: any problem (Ollama off, timeout, bad reply) comes back as
    an AIResult with status 'failed' so the caller can decide what to do.
    """
    model = config.OLLAMA_MODEL
    if not config.ANALYZE_APPEARANCE:
        return AIResult(status="skipped", model=model)

    started = time.time()
    raw = ""
    try:
        payload = {
            "model": model,
            "stream": False,
            "think": bool(config.OLLAMA_THINK),
            "format": OLLAMA_SCHEMA,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {"temperature": 0},
            "messages": [{
                "role": "user",
                "content": OLLAMA_PROMPT,
                "images": [_encode_snapshot(image_path)],
            }],
        }
        req = urllib.request.Request(
            config.OLLAMA_URL.rstrip("/") + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=config.OLLAMA_TIMEOUT) as resp:
            body = json.load(resp)

        raw = (body.get("message") or {}).get("content", "") or ""
        data = json.loads(raw)
        person = bool(data["person_present"])
        confidence = max(0, min(100, int(data["confidence"])))
        latency = int((time.time() - started) * 1000)
        text = _format_analysis(person, confidence, data)
        log.info("AI verdict (%d ms): %s", latency, text.replace("\n", " | "))
        return AIResult("done", person, confidence, text, raw, latency, model)

    except Exception as exc:     # network, timeout, bad JSON, missing key, unreadable image
        latency = int((time.time() - started) * 1000)
        log.warning("AI analysis failed after %d ms: %s", latency, exc)
        return AIResult("failed", raw=raw, latency_ms=latency, model=model, error=str(exc))


# ---------------------------------------------------------------------------
# Notification
# ---------------------------------------------------------------------------

def should_notify(result: AIResult) -> bool:
    """Decide whether this AI result deserves a notification."""
    if result.status == "done":
        return bool(result.person_present) and (
            result.confidence or 0) >= config.PERSON_CONFIDENCE_THRESHOLD
    # 'skipped' (AI turned off) and 'failed' (Ollama problem) fall back to the setting.
    if result.status == "skipped":
        return True
    return bool(config.NOTIFY_ON_AI_FAILURE)


def send_notification(title: str, message: str, image_path: str | None) -> None:
    """Show a desktop toast and beep. Every part is best-effort and never raises."""
    if config.SOUND_ALERT:
        play_alert_sound()
    if not config.NOTIFY_DESKTOP:
        return
    try:
        from winotify import Notification      # Windows only
        icon = os.path.abspath(image_path) if image_path else ""
        toast = Notification(app_id="Security Cam", title=title, msg=message, icon=icon)
        toast.show()
    except ImportError:
        log.warning("winotify not installed (pip install winotify) — no desktop toast.")
    except Exception as exc:
        log.warning("Desktop notification failed: %s", exc)


class LatestText:
    """Thread-safe holder for the newest AI description (shown on the live feed)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._text = ""

    def set(self, text: str) -> None:
        with self._lock:
            self._text = text

    def get(self) -> str:
        with self._lock:
            return self._text


class AnalysisWorker(threading.Thread):
    """Runs the slow AI call off the camera loop.

    The main loop calls submit() and carries on at full frame rate. This thread
    asks Ollama, saves the verdict to the database, and sends the notification.
    """

    def __init__(self, latest: LatestText) -> None:
        super().__init__(name="ai-worker", daemon=True)
        self._queue: queue.Queue = queue.Queue(maxsize=int(config.AI_QUEUE_SIZE))
        self._latest = latest

    def submit(self, event_id: int | None, snapshot_path: str) -> None:
        try:
            self._queue.put_nowait((event_id, snapshot_path))
        except queue.Full:
            log.warning("AI queue full — event %s saved without analysis.", event_id)
            self._finish(event_id, snapshot_path,
                         AIResult("failed", model=config.OLLAMA_MODEL,
                                  error="AI queue full"))

    def stop(self, wait: float = 5.0) -> None:
        self._queue.put(None)
        self.join(timeout=wait)

    def run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            event_id, snapshot_path = item
            self._finish(event_id, snapshot_path, analyze_snapshot(snapshot_path))

    def _finish(self, event_id: int | None, snapshot_path: str, result: AIResult) -> None:
        notify = should_notify(result)
        if result.text:
            self._latest.set(result.text)
        if event_id is not None:
            update_event_ai(event_id, result, notified=notify)
        if notify:
            if result.status == "done":
                title = f"Person detected ({result.confidence}/100)"
            else:
                title = "Movement detected (AI check unavailable)"
            send_notification(title, result.text or "Open the dashboard to review.",
                              snapshot_path)
        else:
            log.info("No notification: AI says person_present=%s confidence=%s (threshold %d).",
                     result.person_present, result.confidence,
                     config.PERSON_CONFIDENCE_THRESHOLD)


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

# Columns added for the Ollama analysis. Databases created by older versions
# get them added automatically by init_db().
AI_COLUMNS = {
    "ai_status":     "TEXT DEFAULT 'pending'",  # pending | done | failed | skipped
    "ai_person":     "INTEGER",                 # 1 person, 0 nobody, NULL unknown
    "ai_confidence": "INTEGER",                 # 0-100
    "ai_model":      "TEXT",
    "ai_latency_ms": "INTEGER",
    "ai_raw":        "TEXT",                    # exact model reply or error text
    "notified":      "INTEGER DEFAULT 0",       # 1 if a notification was sent
    "reviewed":      "INTEGER",                 # human label: 1 correct, 0 false alarm
}


def _connect() -> sqlite3.Connection:
    """Open a new connection. Each thread must use its own."""
    return sqlite3.connect(DB_FILE, timeout=10)


def _db_path(path: str | None) -> str | None:
    """Store paths with forward slashes so they work on any OS."""
    return path.replace("\\", "/") if path else path


def init_db() -> None:
    """Create the events table and add any missing AI columns."""
    with _connect() as conn:
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
        existing = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
        for name, ddl in AI_COLUMNS.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE events ADD COLUMN {name} {ddl}")
                log.info("Database: added column %s", name)
        # WAL lets the admin app read while the camera writes.
        conn.execute("PRAGMA journal_mode=WAL")


def save_event_to_db(
    timestamp: str,
    camera_index: int,
    snapshot_path: str | None,
    video_path: str | None,
    analysis: str,
    box_count: int,
) -> int | None:
    """Insert one detection event and return its row id (None if saving failed)."""
    status = "pending" if config.ANALYZE_APPEARANCE else "skipped"
    try:
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO events "
                "(timestamp, camera_index, snapshot_path, video_path, analysis, "
                " box_count, ai_status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (timestamp, camera_index, _db_path(snapshot_path),
                 _db_path(video_path), analysis, box_count, status),
            )
            return cur.lastrowid
    except sqlite3.Error as exc:
        log.warning("Could not save event to database: %s", exc)
        return None


def update_event_ai(event_id: int, result: "AIResult", notified: bool) -> None:
    """Write the AI verdict (and whether we notified) onto an existing event."""
    person = None if result.person_present is None else int(result.person_present)
    raw = result.raw or result.error
    try:
        with _connect() as conn:
            conn.execute(
                "UPDATE events SET analysis=?, ai_status=?, ai_person=?, "
                "ai_confidence=?, ai_model=?, ai_latency_ms=?, ai_raw=?, notified=? "
                "WHERE id=?",
                (result.text, result.status, person, result.confidence,
                 result.model, result.latency_ms, raw, int(notified), event_id),
            )
    except sqlite3.Error as exc:
        log.warning("Could not update event %s with AI result: %s", event_id, exc)


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
    latest_analysis = LatestText()   # newest AI description, shown on the live feed
    worker = AnalysisWorker(latest_analysis)
    worker.start()

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
                what = "person(s)" if mode == "human" else "motion"
                log.info(
                    "%s confirmed (%d consecutive frames, %d box(es)) — event #%d",
                    what.capitalize(), persist_count, len(boxes), event_count,
                )

                snapshot_path = None
                try:
                    snapshot_path = save_snapshot(frame, config.CAPTURES_DIR)
                except RuntimeError as exc:
                    log.error("Snapshot error: %s", exc)

                if snapshot_path:
                    video_path = recorder.filepath if (recorder and recorder.is_recording) else None

                    # Save the event first, then let the background worker fill
                    # in the AI verdict and send the notification.
                    event_id = save_event_to_db(
                        timestamp=datetime.now().isoformat(timespec="seconds"),
                        camera_index=config.CAMERA_INDEX,
                        snapshot_path=snapshot_path,
                        video_path=video_path,
                        analysis="",
                        box_count=len(boxes),
                    )
                    worker.submit(event_id, snapshot_path)

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
                    analysis=latest_analysis.get(),
                )
                cv2.imshow(config.WINDOW_TITLE, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    log.info("Q pressed — shutting down.")
                    break

    except KeyboardInterrupt:
        log.info("Interrupted — shutting down.")

    finally:
        worker.stop()
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
