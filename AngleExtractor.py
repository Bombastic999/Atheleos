"""
angle_extractor.py — Uses the NEW MediaPipe API (PoseLandmarker)
=================================================================
Google deprecated mp.solutions.pose. This uses the new Tasks API.

Install:
    pip install mediapipe opencv-python numpy

First run will auto-download the pose model (~7MB) into models/pose_landmarker.task

Controls:
    Q — quit
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import urllib.request

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image, ImageFormat

# ── Download model if not present ───────────────────────────────────────────
MODEL_PATH = "models/pose_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

os.makedirs("models", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe pose model (~7MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.\n")


# ── Landmark index reference ─────────────────────────────────────────────────
#  0  nose
# 11  left_shoulder    12  right_shoulder
# 13  left_elbow       14  right_elbow
# 15  left_wrist       16  right_wrist
# 23  left_hip         24  right_hip
# 25  left_knee        26  right_knee
# 27  left_ankle       28  right_ankle


# ── Core angle function ──────────────────────────────────────────────────────
def get_angle(landmarks, a: int, b: int, c: int) -> float:
    """
    Angle at joint B formed by points A-B-C.
    landmarks: list of NormalizedLandmark from the new API
    Returns angle in degrees (0-180).
    """
    A = np.array([landmarks[a].x, landmarks[a].y])
    B = np.array([landmarks[b].x, landmarks[b].y])
    C = np.array([landmarks[c].x, landmarks[c].y])

    v1 = A - B
    v2 = C - B

    cos_angle = np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    )
    return round(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))), 1)


def joints_visible(landmarks, *indices: int, threshold: float = 0.5) -> bool:
    """Returns True if all requested landmarks have visibility above threshold."""
    return all(
        (landmarks[i].visibility or 0) > threshold
        for i in indices
    )


# ── Joint definitions per exercise ──────────────────────────────────────────
EXERCISE_JOINTS = {
    "squat": {
        "left_knee":  (23, 25, 27),   # left_hip  -> left_knee  -> left_ankle
        "right_knee": (24, 26, 28),   # right_hip -> right_knee -> right_ankle
        "left_hip":   (11, 23, 25),   # left_shoulder -> left_hip -> left_knee
        "back_tilt":  (11, 23, 24),   # left_shoulder -> left_hip -> right_hip
    },
    "pushup": {
        "left_elbow":    (11, 13, 15),
        "right_elbow":   (12, 14, 16),
        "left_hip":      (11, 23, 25),   # body line sag check
        "left_shoulder": (13, 11, 23),   # torso angle
    },
    "deadlift": {
        "left_hip":  (11, 23, 25),
        "left_knee": (23, 25, 27),
        "back_tilt": (11, 23, 24),
    },
    "pullup": {
        "left_elbow":    (11, 13, 15),
        "right_elbow":   (12, 14, 16),
        "left_shoulder": (13, 11, 23),
    },
    "lunge": {
        "front_knee": (23, 25, 27),
        "back_knee":  (24, 26, 28),
        "torso":      (11, 23, 25),
    },
    "tricep_dip": {
        "left_elbow":  (11, 13, 15),
        "right_elbow": (12, 14, 16),
        "torso_lean":  (13, 11, 23),
    },
    "plank": {
        "left_hip":  (11, 23, 25),
        "right_hip": (12, 24, 26),
        "left_knee": (23, 25, 27),
    },
    "wall_sit": {
        "left_knee":  (23, 25, 27),
        "right_knee": (24, 26, 28),
        "left_hip":   (11, 23, 25),
        "back_tilt":  (11, 23, 24),
    },
    "crunch": {
        "torso_curl": (11, 23, 25),
        "neck":       (0,  11, 23),
    },
}

# ── Change this to switch exercise ──────────────────────────────────────────
EXERCISE = "squat"
JOINTS   = EXERCISE_JOINTS[EXERCISE]


# ── Draw skeleton manually (new API has no drawing_utils) ────────────────────
POSE_CONNECTIONS = [
    (11,12),(11,13),(12,14),(13,15),(14,16),  # arms
    (11,23),(12,24),(23,24),                   # torso
    (23,25),(24,26),(25,27),(26,28),           # legs
    (27,29),(28,30),(29,31),(30,32),           # feet
]

def draw_skeleton(frame, landmarks, h, w):
    for a, b in POSE_CONNECTIONS:
        if a >= len(landmarks) or b >= len(landmarks):
            continue
        if (landmarks[a].visibility or 0) < 0.4:
            continue
        ax = int(landmarks[a].x * w)
        ay = int(landmarks[a].y * h)
        bx = int(landmarks[b].x * w)
        by = int(landmarks[b].y * h)
        cv2.line(frame, (ax, ay), (bx, by), (0, 180, 255), 2)
    for lm in landmarks:
        if (lm.visibility or 0) < 0.4:
            continue
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)


def draw_angle_label(frame, landmarks, joint_name, a, b, c, h, w):
    if not joints_visible(landmarks, a, b, c):
        return None
    angle = get_angle(landmarks, a, b, c)
    bx = int(landmarks[b].x * w)
    by = int(landmarks[b].y * h)
    cv2.putText(frame, f"{joint_name}: {angle:.0f}deg",
                (bx + 8, by - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 0), 2)
    cv2.circle(frame, (bx, by), 6, (0, 230, 0), -1)
    return angle


# ── Build PoseLandmarker ─────────────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False,
)
landmarker = mp_vision.PoseLandmarker.create_from_options(options)


# ── Main loop ────────────────────────────────────────────────────────────────
def main():
    # Change 0 to "video.mp4" to process a file instead of webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam. Try VideoCapture(1) if you have multiple cameras.")
        return

    print(f"Running  |  exercise: {EXERCISE}")
    print(f"Joints   : {list(JOINTS.keys())}")
    print("Press Q to quit\n")

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame_index += 1

        # New API requires timestamp in milliseconds
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms == 0:
            # fallback for webcams that return 0
            timestamp_ms = frame_index * 33

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]   # first person only

            draw_skeleton(frame, landmarks, h, w)

            current_angles = {}
            for joint_name, (a, b, c) in JOINTS.items():
                angle = draw_angle_label(frame, landmarks, joint_name, a, b, c, h, w)
                if angle is not None:
                    current_angles[joint_name] = angle

            # Print angles to terminal every ~1 second (30 frames)
            if frame_index % 30 == 0 and current_angles:
                print("  " + " | ".join(f"{k}: {v:.0f}" for k, v in current_angles.items()))

        else:
            cv2.putText(frame, "No person detected",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"Exercise: {EXERCISE}  |  Q to quit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Angle Extractor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Done.")


if __name__ == "__main__":
    main()
