from collections import deque
import numpy as np
import json
import cv2

# Constants
GREEN          = (50, 205, 50)
ORANGE         = (0, 165, 255)
RED            = (50, 50, 220)
WHITE          = (240, 240, 240)
GRAY           = (128, 128, 128)
NECK_COLOR     = (255, 0, 0)
BODY_COLOR     = (0, 255, 255)
SHOULDER_COLOR = (0, 255, 0)
LEG_COLOR      = (255, 0, 255)

# Load thresholds and advice
with open('thresholds.json', 'r') as f:
    THRESHOLDS = json.load(f)
with open('advice.json', 'r') as f:
    ADVICE = json.load(f)

# Smoothen the angles
SMOOTH_N = 5
angle_buffer = {
    'torso_lean':   deque(maxlen=SMOOTH_N),
    'head_forward': deque(maxlen=SMOOTH_N),
    'neck_flex':    deque(maxlen=SMOOTH_N),
    'shoulder_sym': deque(maxlen=SMOOTH_N),
    'ear_offset':   deque(maxlen=SMOOTH_N),
}

def smooth(key, value):
    angle_buffer[key].append(value)
    return np.mean(angle_buffer[key])

def get_status(key, value):
    t = THRESHOLDS[key]
    if value <= t['good']:
        return GREEN, 'Good', None
    elif value <= t['warn']:
        return ORANGE, 'Warning', ADVICE[key]['warn']
    else:
        return RED, 'Poor', ADVICE[key]['poor']

def calculate_angle(point1, point2, vertex):
    try:
        a = np.array(point1)
        b = np.array(vertex)
        c = np.array(point2)
        ba = a - b
        bc = c - b
        if np.all(ba == 0) or np.all(bc == 0):
            return 0
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    except Exception as e:
        print(f"Angle error: {e}")
        return 0

def calculate_midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def overlay_posture_angles(image_bgr, pose_landmarks):
    if pose_landmarks is None:
        return image_bgr, None

    h, w = image_bgr.shape[:2]

    def to_px(i):
        lm = pose_landmarks[i]
        # landmark is not visible
        if lm.visibility < 0.05:
            return None
        return (int(lm.x * w), int(lm.y * h))

    try:
        nose           = to_px(0)
        left_ear       = to_px(7)
        right_ear      = to_px(8)
        left_shoulder  = to_px(11)
        right_shoulder = to_px(12)
        left_hip       = to_px(23)
        right_hip      = to_px(24)
        left_knee      = to_px(25)
        right_knee     = to_px(26)

        # verify nose and should landmarks are present
        if any(pt is None for pt in [nose, left_shoulder, right_shoulder]):
            return image_bgr, None
        neck_mid = calculate_midpoint(left_shoulder, right_shoulder)
        ear_mid  = calculate_midpoint(left_ear, right_ear) if left_ear and right_ear else neck_mid
        neck_up  = (neck_mid[0], neck_mid[1] - 100)

        # calculate hip metrics
        has_hips = left_hip is not None and right_hip is not None
        if has_hips:
            hip_mid  = calculate_midpoint(left_hip, right_hip)
            hip_up   = (hip_mid[0], hip_mid[1] - 100)
            hip_down = (hip_mid[0], hip_mid[1] + 100)
            torso_raw = calculate_angle(hip_up, neck_mid, hip_mid)
            cv2.line(image_bgr, left_hip, right_hip, BODY_COLOR, 2)
            cv2.line(image_bgr, hip_mid, neck_mid, BODY_COLOR, 2)
        else:
            torso_raw = 0  # skip torso metric
        
        # calculate knee metrics
        if left_knee and right_knee:
            knee_mid    = calculate_midpoint(left_knee, right_knee)
            leg_angle   = calculate_angle(hip_down, knee_mid, hip_mid)
            is_standing = leg_angle < 50
            cv2.line(image_bgr, hip_mid, knee_mid, LEG_COLOR, 2)
        else:
            knee_mid = 0
            leg_angle   = 0
            is_standing = False

        head_raw    = calculate_angle(neck_up, nose, neck_mid)
        neck_angle  = calculate_angle(nose, hip_mid, neck_mid)
        neck_raw    = max(0.0, 180.0 - neck_angle)
        sym_raw     = abs(left_shoulder[1] - right_shoulder[1]) / h * 100
        ear_raw     = abs(ear_mid[0] - neck_mid[0]) / w * 100

        torso   = smooth('torso_lean',   torso_raw)
        head    = smooth('head_forward', head_raw)
        neck    = smooth('neck_flex',    neck_raw)
        sym     = smooth('shoulder_sym', sym_raw)
        ear_off = smooth('ear_offset',   ear_raw)

        metrics = {
            'torso_lean':   torso,
            'head_forward': head,
            'neck_flex':    neck,
            'shoulder_sym': sym,
            'ear_offset':   ear_off,
            'is_standing':  is_standing,
            'leg_angle':    leg_angle,
        }

        # Draw the Skeleton lines
        cv2.line(image_bgr, left_shoulder, right_shoulder, SHOULDER_COLOR, 2)
        cv2.line(image_bgr, neck_mid, neck_up, GRAY, 1, cv2.LINE_AA)
        cv2.line(image_bgr, neck_mid, nose, NECK_COLOR, 2)
        cv2.circle(image_bgr, neck_mid, 5, GREEN, -1)
        if has_hips:
            cv2.line(image_bgr, left_hip, right_hip, BODY_COLOR, 2)
            cv2.line(image_bgr, hip_mid, neck_mid, BODY_COLOR, 2)
            cv2.line(image_bgr, hip_mid, hip_up, GRAY, 1, cv2.LINE_AA)
            cv2.circle(image_bgr, hip_mid, 5, RED, -1)
        if left_knee and right_knee and has_hips:
            cv2.line(image_bgr, hip_mid, knee_mid, LEG_COLOR, 2)
        if left_ear and right_ear:
            cv2.line(image_bgr, neck_mid, ear_mid, ORANGE, 2)

        # Define metrics to display
        metrics_display = [
        ('neck_flex',    'Neck',       f'{neck:.0f} deg'),
        ('head_forward', 'Head',       f'{head:.0f} deg'),
        ('shoulder_sym', 'Shoulders',  f'{sym:.0f} %'),
        ('ear_offset',   'Ear offset', f'{ear_off:.0f} %'),
        ]
        if has_hips:
            metrics_display.insert(0, ('torso_lean', 'Torso', f'{torso:.0f} deg'))

        # Draw the metrics panel
        panel_x = w - 480
        panel_y = 20
        row_h   = 65
        panel_h = 40 + len(metrics_display) * row_h
        cv2.rectangle(image_bgr, (panel_x - 15, panel_y),
            (w - 10, panel_y + panel_h), (20, 20, 20), -1)
        for i, (key, label, value) in enumerate(metrics_display):
            color, status, advice = get_status(key, metrics[key])
            y = panel_y + 45 + i * row_h
            # colored bar
            cv2.rectangle(image_bgr, (panel_x - 15, y - 28), (panel_x - 6, y + 18), color, -1)
            # metric name
            cv2.putText(image_bgr, label, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
            # value
            cv2.putText(image_bgr, value, (panel_x + 170, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            # advice 
            if advice:
                cv2.putText(image_bgr, f'-> {advice}', (panel_x, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ORANGE, 1, cv2.LINE_AA)
        return image_bgr, metrics
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Overlay error: {e}")
        return image_bgr, None