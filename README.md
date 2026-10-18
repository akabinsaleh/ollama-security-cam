# Security Camera — Local AI Person Detection

Turns a webcam into a security camera that runs entirely on your own PC.
OpenCV finds human-sized movement, then a vision model running in
[Ollama](https://ollama.com) looks at the snapshot and answers "is a real
person there, and how sure are you (0–100)?". A desktop notification fires only
when the answer is yes with enough confidence. No cloud, no API fees.

---

## How it works

```
webcam ─► MOG2 motion + human-size filter ─► snapshot + video clip
                                                   │
                                      row saved in SQLite (status: pending)
                                                   │
                                  background thread asks Ollama (vision model)
                                                   │
              {person_present, confidence 0-100, clothing, appearance, activity}
                                                   │
                     row updated ─► notify if person AND confidence ≥ threshold
```

The AI call runs in a background thread, so the camera keeps recording at full
speed even when the model needs many seconds per image.

## Features

| Feature | Detail |
|---|---|
| Motion detection | MOG2 background subtraction + human-size contour filter |
| False-positive suppression | Area/height/aspect limits + multi-frame persistence |
| Local AI verdict | Ollama vision model returns person yes/no + 0–100 confidence |
| Notification | Windows toast with the snapshot, plus a beep (both optional) |
| Fail-safe | If Ollama is down or too slow, you are still notified (configurable) |
| Snapshot + video | JPEG and an AVI clip with a few seconds of pre-roll |
| Event history | SQLite `events.db`, viewable in the admin dashboard |
| Review labels | Mark events "Real person" / "False alarm" to measure accuracy |
| Cooldown | Minimum gap between events (default 45 s) |

---

## Quick start

1. Install Python 3.10+ and [Ollama](https://ollama.com).
2. Pull a model that has the **vision** capability and check it:
   ```bash
   ollama pull qwen3.5:4b
   ollama show qwen3.5:4b      # "Capabilities" must list vision
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the camera (leave Ollama running):
   ```bash
   python security_cam.py
   ```
5. Review events in the dashboard:
   ```bash
   python admin_app.py
   ```

Press **Q** in the camera window, or **Ctrl+C** in the terminal, to stop.

The first AI answer is slow because Ollama loads the model. Later answers are
faster because `OLLAMA_KEEP_ALIVE` keeps it in memory.

---

## Configuration reference (`config.py`)

### Detection

| Setting | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam index |
| `DETECTION_MODE` | `"human"` | `"human"` filters by blob size/shape, `"motion"` alerts on anything |
| `MIN_HUMAN_HEIGHT` / `MIN_HUMAN_AREA` / `MAX_HUMAN_ASPECT` | `160` / `18000` / `2.0` | Size and shape of a "human-sized" blob |
| `MOTION_PERSIST_FRAMES` | `3` | Consecutive frames needed before an event fires |
| `ALERT_COOLDOWN_SECONDS` | `45` | Minimum gap between events |

### Local AI and notification

| Setting | Default | Description |
|---|---|---|
| `ANALYZE_APPEARANCE` | `True` | Ask Ollama about each snapshot. `False` notifies on every detection |
| `OLLAMA_URL` | `http://localhost:11434` | Where Ollama listens |
| `OLLAMA_MODEL` | `qwen3.5:4b` | Must have the vision capability |
| `OLLAMA_TIMEOUT` | `180` | Seconds to wait for one answer |
| `OLLAMA_KEEP_ALIVE` | `"30m"` | How long the model stays loaded |
| `OLLAMA_THINK` | `False` | `False` skips thinking mode, much faster |
| `OLLAMA_IMAGE_WIDTH` | `640` | Snapshot is shrunk to this width before sending |
| `PERSON_CONFIDENCE_THRESHOLD` | `60` | Notify only at or above this 0–100 score |
| `NOTIFY_ON_AI_FAILURE` | `True` | Still notify when Ollama fails or times out |
| `NOTIFY_DESKTOP` | `True` | Windows toast (needs `winotify`) |
| `SOUND_ALERT` | `True` | Beep when a notification fires (Windows) |
| `AI_QUEUE_SIZE` | `8` | Max snapshots waiting for the AI |

Small models are poor at judging their own certainty, so treat the score as a
rough guide and choose the threshold from your own test results (below).

---

## Data storage

Images and videos stay on disk in `captures/`. The database stores only their
paths (forward slashes) plus the metadata. `events.db` is SQLite in WAL mode so
the dashboard can read while the camera writes.

| Column | Meaning |
|---|---|
| `timestamp`, `camera_index`, `box_count` | When, which camera, how many human-sized blobs |
| `snapshot_path`, `video_path` | Files in `captures/` |
| `analysis` | Readable AI summary |
| `ai_status` | `pending`, `done`, `failed` or `skipped` |
| `ai_person`, `ai_confidence` | AI verdict: 1/0 and 0–100 |
| `ai_model`, `ai_latency_ms`, `ai_raw` | Model name, answer time, exact reply or error |
| `notified` | 1 if a notification was sent |
| `reviewed` | Your label: 1 real person, 0 false alarm |

Databases from older versions get the new columns automatically on the next run.

---

## Measuring accuracy

1. Let the camera collect events, including some with nobody (shadows, a door
   opening) and some with people.
2. Open the dashboard, select each event, and press **Real person** or
   **False alarm**.
3. Count how often the AI agreed with you at a given threshold:
   ```bash
   sqlite3 events.db "
     SELECT (ai_person = 1 AND ai_confidence >= 60) AS ai_says_person,
            reviewed AS you_say_person,
            COUNT(*) AS events
     FROM events
     WHERE reviewed IS NOT NULL AND ai_status = 'done'
     GROUP BY 1, 2;"
   ```
   Change `60` to try other thresholds. Rows where the two columns differ are
   the mistakes: AI says yes and you say no is a false alarm, the reverse is a
   missed person.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Cannot open camera" | Close Zoom/Teams; try `CAMERA_INDEX = 1` |
| `AI analysis failed … Connection refused` | Start Ollama (`ollama serve`) |
| `AI analysis failed … timed out` | Raise `OLLAMA_TIMEOUT`, lower `OLLAMA_IMAGE_WIDTH`, or use a smaller model |
| Every answer fails with an HTTP 400 | The model has no vision capability; run `ollama show <model>` |
| No desktop toast | `pip install winotify`; check Windows notification/focus-assist settings |
| Too many false alarms | Raise `PERSON_CONFIDENCE_THRESHOLD`, `MIN_HUMAN_AREA` or `MOTION_PERSIST_FRAMES` |
| Real people ignored | Lower the threshold, or lower `MIN_HUMAN_AREA` / `MIN_HUMAN_HEIGHT` |

---

## Privacy

This records people. Tell anyone who is filmed, do not point it at shared or
private spaces without permission, and keep `captures/` and `events.db` out of
version control (the included `.gitignore` does this for new files). All
analysis happens locally; nothing is uploaded.

If `events.db`, `motion_events.log` or `config.cpython-314.pyc` were committed
earlier, remove them from tracking without deleting your local copies:

```bash
git rm --cached events.db motion_events.log config.cpython-314.pyc
```

---

## Ideas for later

- Run a small person detector (for example YOLO) before Ollama and compare it
  with the language model on your reviewed events.
- Delete captures older than N days so the disk does not fill.
- Region-of-interest masks so only a doorway is watched.
- Web dashboard or phone notifications through a self-hosted ntfy server.
