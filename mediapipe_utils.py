
import mediapipe as mp
import numpy as np
import cv2
import os

model_path = "pose_landmarker.task"
assert os.path.exists(model_path), f"Missing model: {model_path}"

BaseOptions         = mp.tasks.BaseOptions
PoseLandmarker      = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode   = mp.tasks.vision.RunningMode

with open(model_path, "rb") as f:
    model_data = f.read()

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=model_data),
    running_mode=VisionRunningMode.VIDEO,
)
landmarker = PoseLandmarker.create_from_options(options)
print("Landmarker ready.")


def draw_landmarks_on_image(rgb_image, detection_result, thickness=4, circle_radius=6):
    POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),  # arms
    (11,23),(12,24),(23,24),                   # torso
    (23,25),(25,27),(24,26),(26,28),           # legs
    ]
    annotated = np.copy(rgb_image)
    for pose_landmarks in detection_result.pose_landmarks:
        h, w = annotated.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks]
        for a, b in POSE_CONNECTIONS:
            cv2.line(annotated, pts[a], pts[b], (0, 200, 255), thickness)
        for pt in pts:
            cv2.circle(annotated, pt, circle_radius, (255, 255, 255), -1)
    return annotated