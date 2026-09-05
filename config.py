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
# Play a short beep through the PC speaker when an alert fires.
# Works on Windows only; silently ignored on other platforms.
SOUND_ALERT = True

# --- Event log ---
# Append a timestamped line to a plain-text log file for every alert.
SAVE_EVENT_LOG = True
EVENT_LOG_FILE = "motion_events.log"   # relative to working directory

# --- AI appearance analysis ---
# When True, each snapshot is sent to Claude vision to describe what the
# person is wearing and any other visible details.  The result appears on
# the live feed and is saved to the admin dashboard.
# Requires:  pip install anthropic
ANALYZE_APPEARANCE = True
ANTHROPIC_API_KEY  = "your-anthropic-api-key-here"   # <-- change me
#   Get a key at https://console.anthropic.com/

# --- Display ---
SHOW_LIVE_FEED = True       # Set False to run headless (no window)
WINDOW_TITLE   = "Security Camera — press Q to quit"
