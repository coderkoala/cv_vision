# cv_vision

Live face recognition on an IP camera / NVR stream, using OpenCV's YuNet face detector and SFace
face recogniser. The model weights come from [opencv_zoo](https://github.com/opencv/opencv_zoo),
which is included as a git submodule.

| Script | Purpose |
| --- | --- |
| `facial.py` | Detects faces in the stream and labels each one with the best-matching known person |
| `face.py` | Plain viewer for the RTSP stream, to check the connection |
| `scanner.py` | Tries common RTSP paths on a Xiongmai NVR to find a working stream URL |

## Setup

Requires Python 3.10+, git and [Git LFS](https://git-lfs.com).

```bash
# Clone with the submodule, without downloading every model in opencv_zoo
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/coderkoala/cv_vision.git
cd cv_vision

# Fetch only the two models that are used
git -C opencv_zoo lfs pull --include="models/face_detection_yunet/face_detection_yunet_2023mar.onnx,models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Camera settings

The camera login is read from the environment, so it never ends up in the repository.

```bash
cp .env.example .env    # then edit .env with your camera's details
source .env
```

`facial.py` and `face.py` use `RTSP_URL`. `scanner.py` uses `NVR_IP`, `NVR_PASS` and optionally
`NVR_USER` and `NVR_PORT`; run it if you do not know your camera's stream path.

## Known faces

Put reference photos in a `faces/` folder next to the scripts (it is created on first run and is
git-ignored). Each person can be set up in either way, and the two can be mixed:

```
faces/
├── Alice Smith.jpg          # one photo, the file name is the person's name
└── Bob Jones/               # a folder named after the person, with any number of photos
    ├── front.jpg
    └── glasses.png
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`. Use clear, front-facing photos; if a
photo holds several faces, the largest one is used.

The folder is rescanned every 2 seconds while `facial.py` runs, so photos can be added, replaced
or removed without restarting.

## Running

```bash
python facial.py                      # camera from RTSP_URL, opens a window; press q to quit
python facial.py --no-display         # print matches to the terminal instead
python facial.py --source 0           # webcam
python facial.py --source clip.mp4    # video file
python facial.py --faces /path/to/faces
```

A face is labelled with a name when its cosine similarity to a known photo is at least 0.363
(`COSINE_THRESHOLD` in `facial.py`); otherwise it is shown as `UNKNOWN`. The score is shown next
to the label.
