"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        SEMAPHORE TO KEYBOARD                                ║
║                          CONFIGURATION INDEX                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  TUNING THRESHOLDS (most likely to need adjustment)                          ║
║  ─────────────────────────────────────────────────                           ║
║  VISIBILITY_THRESHOLD      line 46   0.0–1.0. Lower = accept less-confident ║
║                                      landmark detections. Raise if ghost     ║
║                                      arms are triggering false positives.    ║
║                                                                              ║
║  STRAIGHT_LIMB_MARGIN      line 47   Degrees of bend allowed before an arm  ║
║                                      is no longer "straight". Raise if your  ║
║                                      arms are rarely detected as pointing.   ║
║                                                                              ║
║  EXTENDED_LIMB_MARGIN      line 48   Ratio: forearm must be at least this    ║
║                                      fraction of upper-arm length. Raise if  ║
║                                      bent elbows are triggering matches.     ║
║                                                                              ║
║  SNAP_TOLERANCE            line 49   Max degrees of error (per arm) allowed  ║
║                                      when snapping raw angles to the nearest ║
║                                      valid semaphore. Raise to be more       ║
║                                      forgiving; lower for stricter matching. ║
║                                                                              ║
║  GESTURE_CONFIRMATION_FRAMES line 50 How many consecutive video frames must  ║
║                                      show the same pose before it fires.     ║
║                                      Raise to reduce accidental triggers;    ║
║                                      lower for faster response.              ║
║                                                                              ║
║  ANGLE_SNAP_STEP           line 51   Granularity of angle rounding (degrees).║
║                                      Keep at 45 to match the semaphore table.║
║                                                                              ║
║  JUMP DETECTION (caps lock trigger)                                          ║
║  ──────────────────────────────────                                          ║
║  JUMP_THRESHOLD            line 53   How far (0.0–1.0, normalised to frame  ║
║                                      height) the hips must rise above the   ║
║                                      rolling baseline to count as a jump.   ║
║                                      Lower = easier to trigger; raise if    ║
║                                      normal movement is causing false fires. ║
║                                                                              ║
║  JUMP_BASELINE_FRAMES      line 54   How many recent frames to average for  ║
║                                      the "standing" hip baseline. Larger =  ║
║                                      slower to adapt to the performer       ║
║                                      moving towards/away from camera.       ║
║                                                                              ║
║  JUMP_COOLDOWN_FRAMES      line 55   Frames to ignore after a jump fires,   ║
║                                      preventing one jump from toggling caps  ║
║                                      lock multiple times.                   ║
║                                                                              ║
║  BOW DETECTION (tap ↔ hold mode toggle)                                     ║
║  ──────────────────────────────────────                                      ║
║  BOW_THRESHOLD             line 57   How much the shoulder-to-hip distance  ║
║                                      must shrink (normalised 0–1) below the ║
║                                      baseline to count as a bow. Lower =    ║
║                                      more sensitive; raise if normal         ║
║                                      posture changes are false-triggering.  ║
║                                                                              ║
║  BOW_BASELINE_FRAMES       line 58   Frames of torso-height history used to ║
║                                      compute the upright baseline. Larger = ║
║                                      slower to adapt to camera distance.    ║
║                                                                              ║
║  BOW_COOLDOWN_FRAMES       line 59   Frames to suppress re-triggering after ║
║                                      a bow fires (~1 s at 30 fps).          ║
║                                                                              ║
║  SPAM_RATE                 line 61   Key presses per second in spam mode.   ║
║                                      10 = one press every 100ms. Raise for  ║
║                                      faster repeat; lower to slow it down.  ║
║                                                                              ║
║  SPAM_DEBOUNCE_FRAMES      line 62   Consecutive frames of no detected pose ║
║                                      before spam stops. Stops a single      ║
║                                      wobbly frame breaking the spam.        ║
║                                                                              ║
║  OUTPUT                                                                      ║
║  ──────                                                                      ║
║  OUTPUT_FILE               line 54   Path of the text file that confirmed    ║
║                                      gestures are written to. Set to None    ║
║                                      to disable file output.                 ║
║                                                                              ║
║  CAMERA / MODEL                                                              ║
║  ──────────────                                                              ║
║  MODEL_PATH                line 57   Path to the MediaPipe pose model file.  ║
║  CAMERA_INDEX              line 58   0 = default webcam, 1 = next camera.   ║
║                                                                              ║
║  CLI FLAGS                                                                   ║
║  ─────────                                                                   ║
║  --type  /  -t             Actually send keypresses (default: display only)  ║
║  --tolerance N             Override SNAP_TOLERANCE at runtime                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import distance as dist
from math import atan2, degrees

# ── TUNING THRESHOLDS ─────────────────────────────────────────────────────────
VISIBILITY_THRESHOLD       = 0.5   # Min landmark confidence to be considered visible
STRAIGHT_LIMB_MARGIN       = 20    # Max degrees of bend for a "straight" arm (degrees)
EXTENDED_LIMB_MARGIN       = 0.8   # Forearm must be >= this fraction of upper-arm length
SNAP_TOLERANCE             = 15    # Max per-arm angular error for semaphore snapping (degrees)
GESTURE_CONFIRMATION_FRAMES = 5    # Consecutive frames required to confirm a gesture
ANGLE_SNAP_STEP            = 45    # Angle grid size — keep at 45 to match semaphore table

# ── JUMP DETECTION ────────────────────────────────────────────────────────────
JUMP_THRESHOLD       = 0.06  # Hip rise (normalised 0–1) above baseline to count as a jump
JUMP_BASELINE_FRAMES = 20    # Frames of hip history used to compute the standing baseline
JUMP_COOLDOWN_FRAMES = 30    # Frames to suppress further jumps after one fires (~1 s at 30 fps)

# ── BOW DETECTION ─────────────────────────────────────────────────────────────
BOW_THRESHOLD       = 0.08   # Torso-height shrinkage (normalised 0–1) required to count as a bow
BOW_BASELINE_FRAMES = 20     # Frames of torso-height history for the upright baseline
BOW_COOLDOWN_FRAMES = 40     # Frames to suppress re-triggering after a bow fires (~1.3 s at 30 fps)

# ── SPAM MODE ─────────────────────────────────────────────────────────────────
SPAM_RATE           = 10     # Key presses per second when in spam/hold mode
SPAM_DEBOUNCE_FRAMES = 8     # Consecutive frames of no gesture before spam stops.
                             # Prevents a single wobbly frame from interrupting the spam.

# ── OUTPUT ────────────────────────────────────────────────────────────────────
OUTPUT_FILE = "semaphore_output.txt"   # Set to None to disable file logging

# ── CAMERA / MODEL ────────────────────────────────────────────────────────────
MODEL_PATH   = 'pose_landmarker_heavy.task'
CAMERA_INDEX = 0

# ── SKELETON CONNECTIONS (landmark index pairs) ───────────────────────────────
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),   # Left arm
    (12, 14), (14, 16),             # Right arm
    (11, 23), (12, 24), (23, 24),   # Torso
    (23, 25), (25, 27),             # Left leg
    (24, 26), (26, 28),             # Right leg
]

# ── SEMAPHORE LOOKUP TABLE ────────────────────────────────────────────────────
# Keys are (left_arm_angle, right_arm_angle) in degrees, snapped to 45° grid.
# 'a' = alphabetic output, 'n' = numeric/symbol output (when number-shift active).
SEMAPHORES = {
    (-90, -45): {'a': "a",       'n': "1"},
    (-90,   0): {'a': "b",       'n': "2"},
    (-90,  45): {'a': "c",       'n': "3"},
    (-90,  90): {'a': "d",       'n': "4"},
    (135, -90): {'a': "e",       'n': "5"},
    (180, -90): {'a': "f",       'n': "6"},
    (225, -90): {'a': "g",       'n': "7"},
    (-45,   0): {'a': "h",       'n': "8"},
    (-45,  45): {'a': "i",       'n': "9"},
    (180,  90): {'a': "j",       'n': "capslock"},
    ( 90, -45): {'a': "k",       'n': "0"},
    (135, -45): {'a': "l",       'n': "\\"},
    (180, -45): {'a': "m",       'n': "["},
    (225, -45): {'a': "n",       'n': "]"},
    (  0,  45): {'a': "o",       'n': ","},
    ( 90,   0): {'a': "p",       'n': ";"},
    (135,   0): {'a': "q",       'n': "="},
    (180,   0): {'a': "r",       'n': "-"},
    (225,   0): {'a': "s",       'n': "."},
    ( 90,  45): {'a': "t",       'n': "`"},
    (135,  45): {'a': "u",       'n': "/"},
    (225,  90): {'a': "v",       'n': '"'},
    (135, 180): {'a': "w"},
    (135, 225): {'a': "x",       'n': ""},
    (180,  45): {'a': "y"},
    (180, 225): {'a': "z"},
    ( 90,  90): {'a': "space",   'n': "enter"},
    (135,  90): {'a': "tab"},
    (225,  45): {'a': "escape"},
}

# ── GESTURE CONFIRMATION STATE ────────────────────────────────────────────────
gesture_buffer         = []    # Rolling list of detected gesture labels
last_confirmed_gesture = None  # Prevents the same gesture firing twice in a row

# ── CAPS LOCK / JUMP STATE ────────────────────────────────────────────────────
caps_lock_active = False         # Whether caps lock is currently on
hip_y_history    = []            # Rolling buffer of recent hip Y positions (flipped coords)
jump_cooldown    = 0             # Counts down after a jump fires to prevent re-triggering

# ── SPAM MODE / BOW STATE ─────────────────────────────────────────────────────
hold_mode_active       = False   # False = tap mode (one press per gesture confirm)
                                 # True  = spam mode (repeated presses while pose is held)
spam_key               = None    # The key currently being spammed, or None if idle
last_spam_time         = 0.0     # time.time() of the last spam press — used to pace the rate
spam_debounce_counter  = 0       # Counts consecutive frames with no detected gesture.
                                 # Only clears spam_key once this hits SPAM_DEBOUNCE_FRAMES,
                                 # so a single wobbly frame doesn't interrupt the spam.
torso_height_history   = []      # Rolling buffer of shoulder-to-hip distances for bow detection
bow_cooldown           = 0       # Counts down after a bow fires to prevent re-triggering


# ── GEOMETRY HELPERS ──────────────────────────────────────────────────────────

def get_angle(a, b, c):
    """Return the angle (degrees) at joint b, formed by points a–b–c."""
    ang = degrees(
        atan2(c['y'] - b['y'], c['x'] - b['x']) -
        atan2(a['y'] - b['y'], a['x'] - b['x'])
    )
    return ang + 360 if ang < 0 else ang


def get_limb_direction(arm):
    """
    Return the arm's direction snapped to the nearest ANGLE_SNAP_STEP grid.
    arm is a 3-tuple: (shoulder, elbow, wrist) landmark dicts.
    Returns an integer angle; 270° is remapped to -90° for consistency.
    """
    dy = arm[2]['y'] - arm[0]['y']
    dx = arm[2]['x'] - arm[0]['x']
    angle = degrees(atan2(dy, dx))

    # Snap to nearest grid step
    mod_close = angle % ANGLE_SNAP_STEP
    angle -= mod_close
    if mod_close > ANGLE_SNAP_STEP / 2:
        angle += ANGLE_SNAP_STEP

    angle = int(angle)
    return -90 if angle == 270 else angle


def is_limb_pointing(upper, mid, lower):
    """
    Return True if the limb (upper→mid→lower) is:
      - all landmarks sufficiently visible,
      - roughly straight (within STRAIGHT_LIMB_MARGIN degrees), and
      - forearm at least EXTENDED_LIMB_MARGIN × upper-arm length.
    """
    if any(j['visibility'] < VISIBILITY_THRESHOLD for j in [upper, mid, lower]):
        return False

    straightness = abs(180 - get_angle(upper, mid, lower))
    if straightness >= STRAIGHT_LIMB_MARGIN:
        return False

    u_len = dist.euclidean([upper['x'], upper['y']], [mid['x'], mid['y']])
    l_len = dist.euclidean([lower['x'], lower['y']], [mid['x'], mid['y']])
    return l_len >= EXTENDED_LIMB_MARGIN * u_len


# ── JUMP DETECTION ────────────────────────────────────────────────────────────

def detect_jump(body):
    """
    Track hip Y over time and return True when the performer jumps.

    Coordinate note: body dicts use flipped Y (1 - lm.y), so:
      y = 1.0  → top of frame (highest position)
      y = 0.0  → bottom of frame (lowest position)
    A jump therefore shows as hip_y INCREASING above the rolling baseline.

    Returns True once per jump (gated by JUMP_COOLDOWN_FRAMES).
    """
    global hip_y_history, jump_cooldown

    # Tick down cooldown every frame regardless
    if jump_cooldown > 0:
        jump_cooldown -= 1

    hipL = body[23]
    hipR = body[24]

    # Only use hip positions when both are confidently visible
    if (hipL['visibility'] < VISIBILITY_THRESHOLD or
            hipR['visibility'] < VISIBILITY_THRESHOLD):
        return False

    current_hip_y = (hipL['y'] + hipR['y']) / 2.0
    hip_y_history.append(current_hip_y)

    # Keep buffer to JUMP_BASELINE_FRAMES length
    if len(hip_y_history) > JUMP_BASELINE_FRAMES:
        hip_y_history.pop(0)

    # Need a full baseline window before we can reliably detect jumps
    if len(hip_y_history) < JUMP_BASELINE_FRAMES:
        return False

    # Baseline = average of the older frames (exclude the newest 3 which may
    # already be mid-jump)
    baseline = sum(hip_y_history[:-3]) / len(hip_y_history[:-3])

    if current_hip_y > baseline + JUMP_THRESHOLD and jump_cooldown == 0:
        jump_cooldown = JUMP_COOLDOWN_FRAMES
        return True

    return False


# ── BOW DETECTION ─────────────────────────────────────────────────────────────

def detect_bow(body):
    """
    Detect a bow pose by tracking the vertical distance between shoulders and hips.

    Coordinate note: body dicts use flipped Y (1 - lm.y), so:
      y = 1.0  → top of frame    y = 0.0  → bottom of frame
    When standing upright, shoulders sit well above hips:
      torso_height = avg_shoulder_y - avg_hip_y  →  positive value
    When bowing forward, shoulders drop toward hips:
      torso_height SHRINKS below the rolling baseline.

    Returns True once per bow (gated by BOW_COOLDOWN_FRAMES).
    """
    global torso_height_history, bow_cooldown

    if bow_cooldown > 0:
        bow_cooldown -= 1

    shoulderL, shoulderR = body[11], body[12]
    hipL,      hipR      = body[23], body[24]

    landmarks = [shoulderL, shoulderR, hipL, hipR]
    if any(lm['visibility'] < VISIBILITY_THRESHOLD for lm in landmarks):
        return False

    avg_shoulder_y = (shoulderL['y'] + shoulderR['y']) / 2.0
    avg_hip_y      = (hipL['y']      + hipR['y'])      / 2.0
    current_torso_height = avg_shoulder_y - avg_hip_y

    torso_height_history.append(current_torso_height)
    if len(torso_height_history) > BOW_BASELINE_FRAMES:
        torso_height_history.pop(0)

    if len(torso_height_history) < BOW_BASELINE_FRAMES:
        return False

    # Baseline from older frames; exclude newest 3 which may already be mid-bow
    baseline = sum(torso_height_history[:-3]) / len(torso_height_history[:-3])

    if current_torso_height < baseline - BOW_THRESHOLD and bow_cooldown == 0:
        bow_cooldown = BOW_COOLDOWN_FRAMES
        return True

    return False

def snap_to_nearest_semaphore(detected_l_ang, detected_r_ang, tolerance=None):
    """
    Find the closest entry in SEMAPHORES to (detected_l_ang, detected_r_ang).

    Uses SNAP_TOLERANCE (or the override passed in) as the maximum allowed
    error *per arm* — i.e. total distance budget is tolerance * 2.

    Returns (snapped_l_ang, snapped_r_ang, semaphore_dict)
         or (None, None, None) if nothing is close enough.
    """
    if tolerance is None:
        tolerance = SNAP_TOLERANCE

    best_match    = None
    best_distance = float('inf')

    for (l_ang, r_ang), semaphore in SEMAPHORES.items():
        # Angular distance with wrap-around handling
        l_diff = min(abs(detected_l_ang - l_ang), 360 - abs(detected_l_ang - l_ang))
        r_diff = min(abs(detected_r_ang - r_ang), 360 - abs(detected_r_ang - r_ang))
        total  = l_diff + r_diff

        if total < best_distance and total <= tolerance * 2:
            best_distance = total
            best_match    = (l_ang, r_ang)

    if best_match:
        return best_match[0], best_match[1], SEMAPHORES[best_match]

    return None, None, None


# ── GESTURE CONFIRMATION ──────────────────────────────────────────────────────

def check_gesture_confirmation(detected_gesture):
    """
    Accumulate detected_gesture into gesture_buffer.
    Returns the gesture label once GESTURE_CONFIRMATION_FRAMES consecutive
    identical frames are seen (and only when it differs from the last output).
    Returns None otherwise.
    """
    global gesture_buffer, last_confirmed_gesture

    if detected_gesture is None:
        gesture_buffer = []
        return None

    # Reset buffer on gesture change
    if gesture_buffer and gesture_buffer[-1] != detected_gesture:
        gesture_buffer = []

    gesture_buffer.append(detected_gesture)

    if len(gesture_buffer) >= GESTURE_CONFIRMATION_FRAMES:
        if detected_gesture != last_confirmed_gesture:
            last_confirmed_gesture = detected_gesture
            gesture_buffer = []
            return detected_gesture

    return None


# ── OUTPUT ────────────────────────────────────────────────────────────────────

def _resolve_key(key, caps_active):
    """Return the (keyboard_arg, display_string) pair for a key + caps state."""
    if caps_active and len(key) == 1 and key.isalpha():
        return f"shift+{key}", key.upper()
    return key, key


def fire_keypress(kb_key, send_keypress):
    """
    Send a single press_and_release for kb_key.
    Used by both tap mode (once on confirm) and spam mode (called repeatedly).
    """
    if send_keypress:
        import keyboard
        keyboard.press_and_release(kb_key)


def output_gesture(key, frame, send_keypress, log_file, caps_active, hold_mode):
    """
    Handle a newly confirmed gesture.

    TAP MODE  (hold_mode=False):
        Fire a single press_and_release right now. Done.

    SPAM MODE (hold_mode=True):
        Don't press anything here. Instead, store the key in spam_key so the
        main loop can fire it repeatedly at SPAM_RATE times per second.
        The main loop stops spamming when the pose is lost (spam_key → None).
    """
    global spam_key, last_spam_time
    from datetime import datetime

    kb_key, display_key = _resolve_key(key, caps_active)
    timestamp = datetime.now().strftime("%H:%M:%S")
    mode_tag  = "[SPAM]" if hold_mode else "[TAP] "
    print(f"[{timestamp}] ✓ {mode_tag} OUTPUT: '{display_key}'")

    if log_file is not None:
        log_file.write(f"[{timestamp}] {mode_tag} {display_key}\n")
        log_file.flush()

    if hold_mode:
        # Register the key for spam — the main loop ticker will fire it.
        # If the key has changed (e.g. new gesture while already spamming),
        # update spam_key so the new key starts immediately on the next tick.
        if spam_key != kb_key:
            print(f"  → Spam key set to '{kb_key}'")
            spam_key       = kb_key
            last_spam_time = 0.0   # Fire immediately on the very next tick
    else:
        # Tap mode: one clean press_and_release, nothing stored
        fire_keypress(kb_key, send_keypress)
        if not send_keypress:
            # Display-only: draw the key on the frame as visual feedback
            cv2.putText(frame, display_key, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)


# ── DRAWING ───────────────────────────────────────────────────────────────────

def draw_skeleton(image, pose_landmarks):
    """Draw landmark dots and bone lines onto image (in-place)."""
    h, w, _ = image.shape

    for lm in pose_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

    for start_idx, end_idx in POSE_CONNECTIONS:
        s = pose_landmarks[start_idx]
        e = pose_landmarks[end_idx]
        cv2.line(image,
                 (int(s.x * w), int(s.y * h)),
                 (int(e.x * w), int(e.y * h)),
                 (255, 255, 255), 2)


def draw_hud(image, detected_gesture, l_ang, r_ang,
             snapped_l, snapped_r, buffer_count, confirmed,
             caps_active, hold_mode, spam_key_name):
    """Overlay gesture detection info on the camera frame."""
    h, w, _ = image.shape

    # ── Top info panel ────────────────────────────────────────────────────────
    cv2.rectangle(image, (10, 10), (w - 10, 215), (0, 0, 0), -1)
    cv2.rectangle(image, (10, 10), (w - 10, 215), (255, 255, 255), 2)

    y = 50
    if detected_gesture:
        display = detected_gesture.upper() if (caps_active and len(detected_gesture) == 1
                                               and detected_gesture.isalpha()) else detected_gesture
        # Yellow-orange tint when actively spamming, normal green otherwise
        letter_color = (0, 220, 255) if (hold_mode and spam_key_name) else (0, 255, 0)
        cv2.putText(image, f"Letter: {display}", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, letter_color, 3)
        y += 55
        cv2.putText(image, f"Raw:     L={l_ang}°  R={r_ang}°", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        y += 35
        cv2.putText(image, f"Snapped: L={snapped_l}°  R={snapped_r}°", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        y += 35
        bar_color = (0, 255, 0) if buffer_count >= GESTURE_CONFIRMATION_FRAMES else (0, 165, 255)
        cv2.putText(image, f"Confirm: {buffer_count}/{GESTURE_CONFIRMATION_FRAMES}", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)
        if confirmed:
            cv2.putText(image, "CONFIRMED!", (w - 280, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    else:
        cv2.putText(image, "No gesture detected", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

    # ── Banners stack upward from the bottom ──────────────────────────────────
    banner_h = 65
    banner_y = h

    # ── Caps Lock banner ──────────────────────────────────────────────────────
    if caps_active:
        banner_y -= banner_h
        cv2.rectangle(image, (0, banner_y), (w, banner_y + banner_h), (0, 130, 255), -1)
        cv2.rectangle(image, (0, banner_y), (w, banner_y + banner_h), (0, 80, 180), 3)
        _draw_banner_text(image, "CAPS LOCK ON", banner_y, banner_h, w)

    # ── Spam/Hold mode banner — only shown when hold_mode is active ───────────
    if hold_mode:
        banner_y -= banner_h
        if spam_key_name:
            # Actively spamming — bright green to signal the key is firing
            cv2.rectangle(image, (0, banner_y), (w, banner_y + banner_h), (0, 170, 60), -1)
            cv2.rectangle(image, (0, banner_y), (w, banner_y + banner_h), (0, 90, 30), 3)
            _draw_banner_text(image, f"HOLD MODE ON  —  '{spam_key_name}'", banner_y, banner_h, w)
        else:
            # In hold mode but no pose confirmed yet — purple idle
            cv2.rectangle(image, (0, banner_y), (w, banner_y + banner_h), (140, 40, 140), -1)
            cv2.rectangle(image, (0, banner_y), (w, banner_y + banner_h), (80, 0, 80), 3)
            _draw_banner_text(image, "HOLD MODE ON", banner_y, banner_h, w)


def _draw_banner_text(image, label, banner_y, banner_h, w):
    """Draw centred shadowed text inside a banner rectangle."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4
    thickness  = 3
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_x = max(10, (w - text_w) // 2)
    text_y = banner_y + (banner_h + text_h) // 2
    cv2.putText(image, label, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(image, label, (text_x,     text_y),     font, font_scale, (255, 255, 255), thickness)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    global gesture_buffer, last_confirmed_gesture
    global caps_lock_active, hip_y_history, jump_cooldown
    global hold_mode_active, spam_key, last_spam_time, spam_debounce_counter
    global torso_height_history, bow_cooldown

    parser = argparse.ArgumentParser(description="Semaphore-to-keyboard converter")
    parser.add_argument('--type', '-t', action='store_true',
                        help='Actually send keypresses (default: display-only mode)')
    parser.add_argument('--tolerance', type=int, default=SNAP_TOLERANCE,
                        help=f'Angle snap tolerance in degrees (default: {SNAP_TOLERANCE})')
    args = parser.parse_args()

    # Open log file if OUTPUT_FILE is set
    log_file = None
    if OUTPUT_FILE:
        log_file = open(OUTPUT_FILE, 'a', encoding='utf-8')
        from datetime import datetime
        log_file.write(f"\n=== Session started {datetime.now()} ===\n")
        log_file.flush()
        print(f"Logging confirmed gestures to: {OUTPUT_FILE}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_count = 0

        while cap.isOpened():
            cv2.namedWindow('Semaphore', cv2.WINDOW_NORMAL)
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result   = landmarker.detect_for_video(mp_image, timestamp_ms)

            detected_gesture = None
            l_ang = r_ang = snapped_l = snapped_r = None

            if result.pose_landmarks:
                draw_skeleton(frame, result.pose_landmarks[0])

                body = [
                    {'x': 1 - lm.x, 'y': 1 - lm.y, 'visibility': lm.visibility}
                    for lm in result.pose_landmarks[0]
                ]

                # ── Jump → Caps Lock ───────────────────────────────────────
                if detect_jump(body):
                    caps_lock_active = not caps_lock_active
                    state_label = "ON" if caps_lock_active else "OFF"
                    print(f"[{frame_count}] 🔼 JUMP — CAPS LOCK {state_label}")
                    if log_file:
                        from datetime import datetime
                        log_file.write(f"[{datetime.now().strftime('%H:%M:%S')}] [CAPS LOCK {state_label}]\n")
                        log_file.flush()
                    # Caps handled via shift+key; do NOT send a real caps lock keypress.

                # ── Bow → Hold/Tap mode toggle ─────────────────────────────
                if detect_bow(body):
                    hold_mode_active = not hold_mode_active
                    mode_label = "HOLD" if hold_mode_active else "TAP"
                    print(f"[{frame_count}] 🙇 BOW — switched to {mode_label} mode")
                    if log_file:
                        from datetime import datetime
                        log_file.write(f"[{datetime.now().strftime('%H:%M:%S')}] [MODE: {mode_label}]\n")
                        log_file.flush()
                    # When switching back to tap mode, stop any active spam immediately
                    if not hold_mode_active:
                        spam_key              = None
                        spam_debounce_counter = 0

                # ── Arm semaphore detection ────────────────────────────────
                armL = (body[11], body[13], body[15])
                armR = (body[12], body[14], body[16])

                if is_limb_pointing(*armL) and is_limb_pointing(*armR):
                    l_ang = get_limb_direction(armL)
                    r_ang = get_limb_direction(armR)

                    snapped_l, snapped_r, match = snap_to_nearest_semaphore(
                        l_ang, r_ang, tolerance=args.tolerance
                    )

                    if match:
                        detected_gesture = match['a']
                        print(f"[{frame_count}] Gesture: '{detected_gesture}' "
                              f"(L={l_ang}°→{snapped_l}°, R={r_ang}°→{snapped_r}°) "
                              f"| Buffer: {len(gesture_buffer)}/{GESTURE_CONFIRMATION_FRAMES}")
                    else:
                        print(f"[{frame_count}] No semaphore match — "
                              f"L={l_ang}°, R={r_ang}° (tolerance={args.tolerance}°)")
                else:
                    l_vis = all(j['visibility'] >= VISIBILITY_THRESHOLD for j in armL)
                    r_vis = all(j['visibility'] >= VISIBILITY_THRESHOLD for j in armR)
                    if l_vis and r_vis:
                        l_pointing = is_limb_pointing(*armL)
                        r_pointing = is_limb_pointing(*armR)
                        print(f"[{frame_count}] Arms visible but not straight — "
                              f"L_pointing={l_pointing}, R_pointing={r_pointing}")

            # ─────────────────────────────────────────────────────────────────
            # SPAM MODE LOGIC
            # ─────────────────────────────────────────────────────────────────
            # This block runs every frame and has two jobs:
            #
            # JOB 1 — Decide whether to keep spamming or stop.
            #   When a gesture is detected we reset the debounce counter so
            #   the spam keeps going. When the gesture disappears we increment
            #   the counter. Only once it hits SPAM_DEBOUNCE_FRAMES do we
            #   actually clear spam_key and stop. This stops a single wobbly
            #   camera frame from interrupting a sustained hold pose.
            #
            # JOB 2 — Fire the key at the right rate.
            #   We compare the current time against last_spam_time. If enough
            #   time has passed (1 / SPAM_RATE seconds) we fire a single
            #   press_and_release and update last_spam_time. This is entirely
            #   independent of the camera framerate — it will fire at exactly
            #   SPAM_RATE times per second regardless of whether we're running
            #   at 30fps or 60fps.
            # ─────────────────────────────────────────────────────────────────

            confirmed = check_gesture_confirmation(detected_gesture)

            if hold_mode_active:

                # ── JOB 1: Update spam_key based on whether gesture is present ─

                if detected_gesture is not None:
                    # Gesture is visible this frame — reset the debounce counter
                    # so we don't accidentally stop spamming during a brief wobble
                    spam_debounce_counter = 0
                else:
                    # No gesture detected this frame — count up toward the stop threshold
                    spam_debounce_counter += 1
                    if spam_debounce_counter >= SPAM_DEBOUNCE_FRAMES:
                        # Pose has been gone long enough — genuinely lost, stop spam
                        if spam_key is not None:
                            print(f"[{frame_count}] ✋ Pose gone — stopping spam of '{spam_key}'")
                            spam_key = None
                        spam_debounce_counter = 0
                        # Reset so the same gesture can trigger spam again immediately
                        last_confirmed_gesture = None

                # ── JOB 2: Fire the key at SPAM_RATE times per second ─────────

                if spam_key is not None:
                    now           = time.time()
                    spam_interval = 1.0 / SPAM_RATE   # e.g. 0.1 s between presses at 10/s

                    if now - last_spam_time >= spam_interval:
                        # Enough time has elapsed — fire a press_and_release
                        fire_keypress(spam_key, args.type)
                        last_spam_time = now
                        # Only print occasionally to avoid flooding the console
                        if args.type:
                            print(f"  💥 Spam: '{spam_key}'")

            else:
                # ── TAP MODE: normal single-press on confirmation ──────────────
                # Nothing extra to do here; output_gesture handles the press.
                pass

            if confirmed:
                output_gesture(confirmed, frame, args.type, log_file,
                               caps_lock_active, hold_mode_active)

            draw_hud(frame, detected_gesture,
                     l_ang, r_ang, snapped_l, snapped_r,
                     len(gesture_buffer), bool(confirmed),
                     caps_lock_active, hold_mode_active, spam_key)

            cv2.imshow('Semaphore', frame)
            if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
                break

    cap.release()
    cv2.destroyAllWindows()
    if log_file:
        from datetime import datetime
        log_file.write(f"=== Session ended {datetime.now()} ===\n")
        log_file.close()
        print(f"Log saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()