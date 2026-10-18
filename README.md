# facecam

Detect and recognise your face from a webcam, using OpenCV's YuNet (detection) and SFace (recognition) models. Models are downloaded automatically on first run.

## Setup
```
pip install -r requirements.txt
```

## Usage
```
python -m facecam enroll --name Me   # capture 15 samples of your face
python -m facecam run                # live view: green = you, red = Unknown, q quits
python -m facecam image photo.jpg    # check a single photo
```

## Layout
```
facecam/
  cli.py      command-line commands (enroll, run, image) and drawing
  models.py   model download/loading, detect, embed, identify, face database
  config.py   paths, model URLs, thresholds
models/       downloaded models (git-ignored)
data/         enrolled faces, faces.npz (git-ignored)
```

Tune `MATCH_THRESHOLD` in `facecam/config.py`: raise it if strangers match, lower it if you aren't recognised.
