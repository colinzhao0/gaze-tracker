# gaze-tracker
# Gaze Tracker

This project is a Python-based gaze tracking system that uses opencv and [MediaPipe](https://google.github.io/mediapipe/) Face Mesh to estimate where a user is looking and detect blinks in real time.

## Features

- Real-time gaze tracking using webcam video.
- Blink detection using Eye Aspect Ratio (EAR).
- Calibration routine for improved accuracy.
- Visual overlay of gaze point and angles.
- Gaze and blink data logging to `gazetracking_data.csv`.

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)
- SciPy (`scipy`)

Notes:
- Press ESC during calibration to abort.
- The script uses the first detected face in the webcam feed.
- Data is overwritten each run.

Install dependencies with:

```sh
pip install opencv-python mediapipe numpy scipy

