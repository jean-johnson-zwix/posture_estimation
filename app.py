
import gradio as gr
import utils
import time
import numpy as np
import mediapipe_utils
import mediapipe as mp
import cv2

_last_ts_ms = 0

def process_frame(frame):
    global _last_ts_ms

    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    ts_ms = int(time.time_ns() // 1_000_000)
    if ts_ms <= _last_ts_ms:
        ts_ms = _last_ts_ms + 1
    _last_ts_ms = ts_ms

    mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    try:
        result = mediapipe_utils.landmarker.detect_for_video(mediapipe_image, ts_ms)
    except Exception as e:
        print(f"Detection error: {e}")
        return frame  # return raw frame on detection failure

    first_pose = result.pose_landmarks[0] if result.pose_landmarks else None
    first_pose = result.pose_landmarks[0] if result.pose_landmarks else None
    annotated  = mediapipe_utils.draw_landmarks_on_image(frame, result, thickness=5, circle_radius=8)
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    result_tuple = utils.overlay_posture_angles(annotated_bgr, first_pose)
    if result_tuple is None or result_tuple[0] is None:
        # overlay failed — return annotated without posture overlay
        return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    annotated_bgr, metrics = result_tuple
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return annotated_rgb


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("Real-Time Posture Analysis")

    with gr.Row():
        with gr.Column(scale=1):
            cam = gr.Image(
                label="Webcam",
                sources="webcam",
                type="numpy",
                streaming=True,
                height=240
            )
        with gr.Column(scale=3):
            out_img = gr.Image(
                label="Posture Analysis",
                type="numpy",
                height=600
            )

    cam.stream(
        fn=process_frame,
        inputs=cam,
        outputs=[out_img],
        time_limit=120,
        stream_every=0.25,
        concurrency_limit=1
    )

demo.launch()