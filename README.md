# ASL Keyboard

A real-time American Sign Language (ASL) keyboard that uses your webcam to recognize hand signs and convert them into typed text. Hold up an ASL letter, hit Enter to place it, and build words letter by letter.

---

## Features

- Real-time hand detection using MediaPipe
- ML model classifies static ASL letters (A–Y, excluding J and Z)
- Motion-based detection for J and Z using index fingertip velocity
- On-screen display of current prediction and typed text
- DELETE and SPACE gesture support
- Works entirely offline after setup

---

## How It Works

The app uses two systems working together:

**Static letters (A–I, K–Y)** are recognized by a trained ML model. Each frame, MediaPipe extracts 21 hand landmarks (x, y, z coordinates = 63 values), which are fed into the model to predict the letter.

**Motion letters (J and Z)** are detected by tracking the index fingertip position between frames. If the hand is in the I pose and moving, the prediction becomes J. If it's in the D pose and moving, the prediction becomes Z.

---

## Requirements

- Python 3.8+
- Webcam
- macOS / Windows / Linux

Install dependencies:

```bash
pip install opencv-python mediapipe numpy scikit-learn
```

> **macOS users:** Make sure your terminal has camera access enabled in **System Settings → Privacy & Security → Camera**

---

## Setup

You will need to train your own model on your hand data before running the keyboard. The model and label encoder should be saved and loaded in the main script:

```python
import joblib
model = joblib.load("asl_model.pkl")
encoder = joblib.load("asl_encoder.pkl")
```

Make sure your model accepts a `(1, 63)` input array of MediaPipe landmark coordinates and returns a class prediction that your encoder can inverse transform into a letter string.

---

## Usage

Run the keyboard:

```bash
python asl_keyboard.py
```

| Action | How |
|---|---|
| Preview a letter | Hold up the ASL sign — prediction appears on screen |
| Type the letter | Press **Enter** |
| Add a space | Hold SPACE sign → press **Enter** |
| Delete last letter | Hold DELETE sign → press **Enter** |
| Quit | Press **Esc** |

---


---

## Project Structure

```
asl_keyboard.py      # Main application
asl_model.pkl        # Trained ML model (you provide)
asl_encoder.pkl      # Label encoder (you provide)
README.md
```

---

## Troubleshooting

**Camera won't open**
- Check camera permissions 
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` if you have multiple cameras

**J or Z not detecting**
- Adjust `MOVEMENT_THRESHOLD` in the script (lower = more sensitive, higher = less sensitive)

**Letters flickering between predictions**
- Hold your hand more still while signing
- Ensure good, consistent lighting on your hand
- Make sure your hand fills a reasonable portion of the frame


---

## Known Limitations

- One hand only
- Best performance in good lighting with a plain background
- J and Z detection sensitivity may need tuning depending on webcam framerate and individual signing style
- Model accuracy could be better with more data collected 
