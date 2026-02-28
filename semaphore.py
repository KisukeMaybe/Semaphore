import argparse
import keyboard
import mediapipe as mp
import cv2
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import distance as dist
from math import atan, atan2, pi, degrees
from datetime import datetime

# --- SETTINGS & CONSTANTS ---
MODEL_PATH = 'pose_landmarker_heavy.task'
RECORDING_FILENAME = str(datetime.now()).replace(
    '.', '').replace(':', '') + '.avi'
FPS = 10
VISIBILITY_THRESHOLD = 0.5  # Tasks API is often more confident at lower numbers
STRAIGHT_LIMB_MARGIN = 20
EXTENDED_LIMB_MARGIN = 0.8
LEG_LIFT_MIN = -30
ARM_CROSSED_RATIO = 2
MOUTH_COVER_THRESHOLD = 0.03
SQUAT_THRESHOLD = 0.1
JUMP_THRESHOLD = 0.0001
LEG_ARROW_ANGLE = 18
FINGER_MOUTH_RATIO = 1.5

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),           # Right arm
    (11, 23), (12, 24), (23, 24),  # Torso
    (23, 25), (25, 27),           # Left leg
    (24, 26), (26, 28)            # Right leg
]

SEMAPHORES = {
    (-90, -45): {'a': "a", 'n': "1"}, (-90, 0): {'a': "b", 'n': "2"},
    (-90, 45): {'a': "c", 'n': "3"}, (-90, 90): {'a': "d", 'n': "4"},
    (135, -90): {'a': "e", 'n': "5"}, (180, -90): {'a': "f", 'n': "6"},
    (225, -90): {'a': "g", 'n': "7"}, (-45, 0): {'a': "h", 'n': "8"},
    (-45, 45): {'a': "i", 'n': "9"}, (180, 90): {'a': "j", 'n': "capslock"},
    (90, -45): {'a': "k", 'n': "0"}, (135, -45): {'a': "l", 'n': "\\"},
    (180, -45): {'a': "m", 'n': "["}, (225, -45): {'a': "n", 'n': "]"},
    (0, 45): {'a': "o", 'n': ","}, (90, 0): {'a': "p", 'n': ";"},
    (135, 0): {'a': "q", 'n': "="}, (180, 0): {'a': "r", 'n': "-"},
    (225, 0): {'a': "s", 'n': "."}, (90, 45): {'a': "t", 'n': "`"},
    (135, 45): {'a': "u", 'n': "/"}, (225, 90): {'a': "v", 'n': '"'},
    (135, 180): {'a': "w"}, (135, 225): {'a': "x", 'n': ""},
    (180, 45): {'a': "y"}, (180, 225): {'a': "z"},
    (90, 90): {'a': "space", 'n': "enter"}, (135, 90): {'a': "tab"},
    (225, 45): {'a': "escape"},
}

# --- GLOBAL STATE ---
frame_midpoint = (0, 0)
current_semaphore = ''
last_keys = []
FRAME_HISTORY = 8
empty_frame = {'hipL_y': 0, 'hipR_y': 0, 'hips_dy': 0, 'signed': False}
last_frames = FRAME_HISTORY * [empty_frame.copy()]

# --- HELPER FUNCTIONS (Logic remains largely same, just fed from new API) ---


def get_angle(a, b, c):
    ang = degrees(atan2(c['y']-b['y'], c['x']-b['x']) -
                  atan2(a['y']-b['y'], a['x']-b['x']))
    return ang + 360 if ang < 0 else ang


def get_limb_direction(arm, closest_degrees=45):
    dy = arm[2]['y'] - arm[0]['y']
    dx = arm[2]['x'] - arm[0]['x']
    angle = degrees(atan2(dy, dx))
    mod_close = angle % closest_degrees
    angle -= mod_close
    if mod_close > closest_degrees/2:
        angle += closest_degrees
    angle = int(angle)
    return -90 if angle == 270 else angle


def is_limb_pointing(upper, mid, lower):
    if any(j['visibility'] < VISIBILITY_THRESHOLD for j in [upper, mid, lower]):
        return False
    is_in_line = abs(180 - get_angle(upper, mid, lower)) < STRAIGHT_LIMB_MARGIN
    if is_in_line:
        u_len = dist.euclidean([upper['x'], upper['y']], [mid['x'], mid['y']])
        l_len = dist.euclidean([lower['x'], lower['y']], [mid['x'], mid['y']])
        return l_len > EXTENDED_LIMB_MARGIN * u_len
    return False


def output(keys, image, display_only=True):
    keystring = '+'.join(keys)
    if len(keystring):
        print("Trigerred:", keystring)
        if not display_only:
            keyboard.press_and_release(keystring)
        else:
            cv2.putText(image, keystring, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)


def draw_manual_skeleton(image, pose_landmarks):
    h, w, _ = image.shape

    # 1. Draw the joints (Circles)
    for lm in pose_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)  # Green dots

    # 2. Draw the bones (Lines)
    for start_idx, end_idx in POSE_CONNECTIONS:
        start_lm = pose_landmarks[start_idx]
        end_lm = pose_landmarks[end_idx]

        # Convert normalized (0-1) to pixel coordinates
        start_point = (int(start_lm.x * w), int(start_lm.y * h))
        end_point = (int(end_lm.x * w), int(end_lm.y * h))

        cv2.line(image, start_point, end_point,
                 (255, 255, 255), 2)  # White lines

# --- MAIN ENGINE ---


def main():
    global last_frames, frame_midpoint, current_semaphore, last_keys

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '-t', action='store_true',
                        help='Actually send keys')
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            cv2.namedWindow('Semaphore Tasks API', cv2.WINDOW_NORMAL)
            success, frame = cap.read()
            if not success:
                break

            # Tasks API requires mp.Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp)

            if result.pose_landmarks:
                # Draw the skeleton
                draw_manual_skeleton(frame, result.pose_landmarks[0])

                # Convert Result to our 'body' format
                body = []
                for landmark in result.pose_landmarks[0]:
                    body.append({'x': 1 - landmark.x, 'y': 1 -
                                landmark.y, 'visibility': landmark.visibility})

                # Simple Gesture Logic (Mouth Cover example)
                mouth = (body[9], body[10])
                palms = (body[19], body[20])
                if abs(mouth[0]['x'] - palms[0]['x']) < MOUTH_COVER_THRESHOLD:
                    output(['backspace'], frame, not args.type)

                # Arm Semaphore Logic
                armL = (body[11], body[13], body[15])
                armR = (body[12], body[14], body[16])
                if is_limb_pointing(*armL) and is_limb_pointing(*armR):
                    l_ang, r_ang = get_limb_direction(
                        armL), get_limb_direction(armR)
                    match = SEMAPHORES.get((l_ang, r_ang))
                    if match:
                        key = match['a']
                        if [key] != last_keys:
                            output([key], frame, not args.type)
                            last_keys = [key]

            cv2.imshow('Semaphore Tasks API', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
