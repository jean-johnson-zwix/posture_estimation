# Real-Time Posture Estimation

A real-time posture analysis tool built with MediaPipe Pose Landmarker and Gradio. 
Detects poor sitting habits and gives actionable feedback — live, on your webcam feed.

---

## Tech Stack

- [MediaPipe Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) — pose estimation
- [OpenCV](https://opencv.org/) — frame annotation
- [Gradio](https://gradio.app/) — webcam UI
- Python 3.10+

---

## Posture Metrics

| Metric | What It Measures |
|---|---|
| **Torso Lean** | How far your back deviates from vertical |
| **Neck Flex** | Downward bend of your neck |
| **Head Forward** | How far your head protrudes in front of your shoulders |
| **Shoulder Symmetry** | Height difference between left and right shoulders |
| **Ear Offset** | Horizontal distance of your ear from your shoulder centerline |


## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/posture-estimation
cd posture-estimation
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the MediaPipe model

```bash
wget -O pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

### 5. Run the app

```bash
gradio on app.py
```