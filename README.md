# Drone Eye-Tracking Project

## Overview

This repository contains Python scripts for eye tracking and face detection, likely intended for drone navigation or similar applications. The project uses machine learning, computer vision libraries, and real-time video processing to achieve its goals.

## Table of Contents

- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
  - [Training the Eye Position Model](#training-the-eye-position-model)
  - [Real-Time Eye Position Prediction](#real-time-eye-position-prediction)
  - [Simple Eye and Face Detection](#simple-eye-and-face-detection)
  - [Eye Labeling](#eye-labeling)
  - [Alternative Eye and Face Detection](#alternative-eye-and-face-detection)
- [Contributing](#contributing)
- [License](#license)

## Dependencies

- Python 3.x
- PyTorch
- OpenCV
- dlib
- PIL (Pillow)
- imutils

You can install these packages using pip:

\```bash
pip install torch torchvision opencv-python dlib Pillow imutils
\```

## How to Run

### Training the Eye Position Model

Run `trainer.py` to train a Convolutional Neural Network (CNN) for eye position classification.

\```bash
python trainer.py
\```

### Real-Time Eye Position Prediction

Run `main.py` to perform real-time eye position prediction using the trained model.

\```bash
python main.py
\```

### Simple Eye and Face Detection

Run `eye_tracking.py` for a simpler version of eye and face detection using OpenCV's Haar Cascades.

\```bash
python eye_tracking.py
\```

### Eye Labeling

Run `eye_labeler.py` to label eye positions in real-time. This will save frames and labels for future use.

\```bash
python eye_labeler.py
\```

### Alternative Eye and Face Detection

Run `claude.py` for an alternative method of eye and face detection using dlib.

\```bash
python claude.py
\```

## Contributing

Feel free to fork this repository, create a new branch, and submit a pull request.

## License

This project is licensed under the MIT License.

