# EmoVision

Real-time facial emotion detection using your webcam. EmoVision finds faces with OpenCV and classifies each one with a small CNN trained on 48×48 grayscale face images.

## Features

- Live webcam feed with a mirrored view
- Face detection using OpenCV Haar cascades
- Recognizes 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise
- Confidence percentage shown above each detected face
- Confidence bars for the main (largest) face
- Smoothing over the last 7 frames so labels don't flicker

## Project Structure

```
EmoVision/
├── live.py            # Runs the webcam app
├── predictor.py       # Face detection + emotion prediction
├── emotion_cnn.keras  # Trained CNN model
├── labels.json        # Emotion class names
└── requirements.txt   # Python dependencies
```

## Setup

1. Clone the repository:

   ```
   git clone https://github.com/Anfel-bth/EmoVision.git
   cd EmoVision
   ```

2. Create and activate a virtual environment (recommended):

   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

   > **Note:** use OpenCV 4.x (`opencv-python<5`). OpenCV 5.x removed `cv2.CascadeClassifier`, which this project relies on.

## Usage

```
python live.py
```

A window opens showing your webcam feed. Press **q** or **Esc** to quit.

## How It Works

1. `predictor.py` converts each frame to grayscale and detects faces with a Haar cascade.
2. Each face is cropped, resized to 48×48 and scaled to 0–1.
3. The CNN outputs a probability for each of the 7 emotions.
4. `live.py` averages the last few frames and draws the results on the video.

## Author

**Anfel Benatmane**
