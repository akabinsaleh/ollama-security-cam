# =============================================================================
# config.py — All tunable settings live here.
# Edit this file before running security_cam.py.
# =============================================================================

# --- Camera ---
CAMERA_INDEX = 0          # 0 = default webcam; try 1, 2 … for external cameras

# --- Camera resilience ---
# If the camera disconnects, retry this many times before giving up.
CAMERA_RECONNECT_ATTEMPTS = 5
CAMERA_RECONNECT_DELAY    = 3.0   # seconds between reconnect attempts

# --- Motion detection ---
# Minimum contour area (in pixels²) that counts as real motion.
# Raise this number to ignore small movements (shadows, pets, leaves).
# Lower it to catch subtler motion.
MOTION_THRESHOLD = 1500

# MOG2 background-subtractor sensitivity (0–255).
# Lower → more sensitive to change; higher → less sensitive.
MOG2_SENSITIVITY = 25

# Gaussian blur kernel size (must be odd).  Reduces noise before differencing.
BLUR_KERNEL = (21, 21)

# Minimum number of frames motion must persist before an alert fires.
# Helps filter out single-frame noise (e.g. a sudden flash of light).
MOTION_PERSIST_FRAMES = 3

# --- Alert cooldown ---
# Minimum seconds between two consecutive email alerts.
# Set to 0 to disable cooldown (not recommended).
ALERT_COOLDOWN_SECONDS = 45

# --- Captures folder ---
CAPTURES_DIR = "captures"   # Relative to the script's working directory

# --- Detection mode ---
# "human"  — alert only on human-sized blobs (recommended).
#            Filters out small animals, insects, leaves, and camera noise
#            while reliably catching people at any angle, partial body, or
#            clothing style (including loose robes / thobes).
# "motion" — alert on any movement regardless of size (legacy).
DETECTION_MODE = "human"

# Human-scale contour filters (used in "human" mode)
#
# MIN_HUMAN_HEIGHT  : bounding-box height in pixels a blob must reach.
#   80 px ≈ a person's lower body filling ~1/9 of a 720 p frame height.
#   Lower this if you need to catch people very far from the camera.
#
# MIN_HUMAN_AREA    : filled pixel area of the blob.
#   5 000 px² rules out hands, small pets, and flickering shadows.
#   A partial lower body in a loose thobe is typically 15 000–60 000 px².
#
# MAX_HUMAN_ASPECT  : max width-to-height ratio.
#   Humans are taller than wide; 2.0 gives headroom for wide clothing.
#   Values above ~2.5 are more likely furniture / animals / passing cars.
MIN_HUMAN_HEIGHT = 160    # px — rules out hands, wrists, small objects on a desk
MIN_HUMAN_AREA   = 18000  # px² — rules out forearms, small animals, shadows
MAX_HUMAN_ASPECT = 2.0

# --- Video recording ---
# When True, a short video clip (.avi) is saved for each motion event
# in addition to the JPEG snapshot.  The clip includes a pre-motion
# "pre-roll" so you see what was happening before the alarm fired.
RECORD_VIDEO = True
VIDEO_PRE_BUFFER_FRAMES   = 60    # frames stored before motion fires (~3 s at 20 fps)
VIDEO_POST_MOTION_SECONDS = 5.0   # keep recording this long after motion stops
VIDEO_FPS                 = 20.0  # playback framerate for saved .avi clips

# --- Sound alert ---
# Play a short beep through the PC speaker when a notification fires.
# Works on Windows only; silently ignored on other platforms.
SOUND_ALERT = True

# --- Event log ---
# Append a timestamped line to a plain-text log file for every alert.
SAVE_EVENT_LOG = True
EVENT_LOG_FILE = "motion_events.log"   # relative to working directory

# --- Local AI analysis (Ollama) ---
# When True, each snapshot is sent to a vision model running in your own
# Ollama install.  Nothing leaves your PC and there is no API cost.
# The model answers: is a real person visible, how sure (0-100), and what
# they look like.  The result appears on the live feed and in the dashboard.
# When False, every confirmed detection notifies you without an AI check.
ANALYZE_APPEARANCE = True

OLLAMA_URL        = "http://localhost:11434"
OLLAMA_MODEL      = "qwen3.5:4b"   # must have the "vision" capability (ollama show <model>)
OLLAMA_TIMEOUT    = 180            # seconds to wait for one answer (CPU-only PCs are slow)
OLLAMA_KEEP_ALIVE = "30m"          # keep the model loaded in memory between events
OLLAMA_THINK      = False          # False = skip "thinking" mode, much faster
OLLAMA_IMAGE_WIDTH = 640           # snapshot is shrunk to this width before sending

# --- Person notification ---
# A notification fires only when the AI says a person is present AND its
# confidence (0-100) is at least this value.  Tune it on your own test data.
PERSON_CONFIDENCE_THRESHOLD = 60

# If Ollama is off, times out, or gives an unreadable answer, still notify
# (a missed intruder is worse than a false alarm).  Set False to stay silent.
NOTIFY_ON_AI_FAILURE = True

# Show a Windows desktop toast notification.  Needs:  pip install winotify
NOTIFY_DESKTOP = True

# Max snapshots waiting for the AI.  If the AI is slower than events arrive,
# extra events are saved and notified without analysis instead of piling up.
AI_QUEUE_SIZE = 8

# --- Display ---
SHOW_LIVE_FEED = True       # Set False to run headless (no window)
WINDOW_TITLE   = "Security Camera — press Q to quit"
