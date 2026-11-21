"""
form_coach.py - small prototype: rep counting + rep speed + form check
=======================================================================

Usage:
    python form_coach.py --exercise squat                 live coaching
    python form_coach.py --exercise squat --record        record training data
    python form_coach.py --exercise squat --source clip.mp4

Form check:
    - If models/<exercise>.pkl exists (made by train.py) the SVM judges each rep.
    - Otherwise simple angle rules (RULES below) are used as a fallback.

Keys:
    Q quit | R recalibrate | 1/2/3 pick label (record mode only)
"""

import argparse
import csv
import os
import pickle
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# -- Tunable settings --------------------------------------------------------
CAL_FRAMES     = 45     # frames to hold still at the start (about 1.5 s)
DOWN_DELTA     = 20     # degrees below start angle = rep has started       
UP_DELTA       = 15     # within this many degrees of start = rep finished
MIN_ROM        = 40     # smaller range of motion = partial rep, not counted
MIN_DOWN_TIME  = 0.8    # seconds; faster descent gets a "too fast" note
VIS_THRESHOLD  = 0.5    # minimum landmark visibility
SMOOTH_WINDOW  = 5      # frames of smoothing on the driver angle

LABEL_KEYS = {ord("1"): "good", ord("2"): "shallow", ord("3"): "bad_posture"}


# -- Landmark indices per body side ------------------------------------------
SIDES = {
    "left":  {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "knee": 25, "ankle": 27},
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24, "knee": 26, "ankle": 28},
}

CONNECTIONS = [
    (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
]


# -- Exercise definitions ----------------------------------------------------
# "features": 3 joint names = angle at the middle joint
#             2 joint names = angle of the line (second -> first) vs vertical
# "driver"  : the feature that drives rep counting
# "rules"   : fallback form checks on the bottom position (absolute angles)
EXERCISES = {
    "squat": {
        "driver": "knee",
        "features": {
            "knee": ("hip", "knee", "ankle"),
            "hip":  ("shoulder", "hip", "knee"),
            "lean": ("shoulder", "hip"),
        },
        "rules": [
            ("knee", "<=", 100, "Go deeper"),
            ("lean", "<=", 55,  "Chest up, leaning too far forward"),
        ],
        "hint": "Stand side-on to the camera, full body in frame",
    },
    "pushup": {
        "driver": "elbow",
        "features": {
            "elbow": ("shoulder", "elbow", "wrist"),
            "body":  ("shoulder", "hip", "ankle"),
        },
        "rules": [
            ("elbow", "<=", 100, "Go lower"),
            ("body",  ">=", 160, "Keep hips in line with shoulders"),
        ],
        "hint": "Camera at the side, start in the top position",
    },
    "curl": {
        "driver": "elbow",
        "features": {
            "elbow": ("shoulder", "elbow", "wrist"),
            "swing": ("hip", "shoulder", "elbow"),
        },
        "rules": [
            ("elbow", "<=", 60, "Curl higher"),
            ("swing", "<=", 35, "Keep your elbow still"),
        ],
        "hint": "Stand side-on, arm hanging straight to start",
    },
}

OPS = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b}


# -- MediaPipe setup ---------------------------------------------------------
MODEL_PATH = "models/pose_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)


def build_landmarker():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe pose model (about 7 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


# -- Geometry (pixel coordinates, so angles are not distorted) ---------------
def get_point(lm, side, name, w, h):
    p = lm[SIDES[side][name]]
    return np.array([p.x * w, p.y * h])


def angle_3pt(a, b, c):
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def angle_vs_vertical(top, bottom):
    v = top - bottom
    cos = np.dot(v, np.array([0.0, -1.0])) / (np.linalg.norm(v) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def compute_features(lm, side, spec, w, h):
    out = {}
    for name, joints in spec["features"].items():
        pts = [get_point(lm, side, j, w, h) for j in joints]
        out[name] = angle_3pt(*pts) if len(pts) == 3 else angle_vs_vertical(*pts)
    return out


def joints_visible(lm, side, spec):
    needed = {j for joints in spec["features"].values() for j in joints}
    return all((lm[SIDES[side][j]].visibility or 0) > VIS_THRESHOLD for j in needed)


def choose_side(lm):
    def score(side):
        return sum((lm[i].visibility or 0) for i in SIDES[side].values())
    return "left" if score("left") >= score("right") else "right"


# -- Rep counter + speed -----------------------------------------------------
class RepCounter:
    def __init__(self, start_angle):
        self.start = start_angle
        self.state = "top"
        self.buf = deque(maxlen=SMOOTH_WINDOW)
        self.smoothed = start_angle
        self.min_angle = start_angle
        self.t_leave = self.t_bottom = 0.0
        self.bottom_feats = {}

    def update(self, angle, feats, t):
        self.buf.append(angle)
        a = float(np.mean(self.buf))
        self.smoothed = a
        event = None

        if self.state == "top":
            if a < self.start - DOWN_DELTA:
                self.state = "down"
                self.t_leave = self.t_bottom = t
                self.min_angle = a
                self.bottom_feats = feats

        else:
            if a < self.min_angle:
                self.min_angle = a
                self.t_bottom = t
                self.bottom_feats = feats
            if a > self.start - UP_DELTA:
                self.state = "top"
                rom = self.start - self.min_angle
                if rom >= MIN_ROM:
                    event = {
                        "rom": rom,
                        "down": self.t_bottom - self.t_leave,
                        "up": t - self.t_bottom,
                        "bottom": self.bottom_feats,
                    }
                else:
                    event = {"partial": True, "rom": rom}
        return a, event


# -- Form judgement ----------------------------------------------------------
def load_form_model(exercise):
    path = f"models/{exercise}.pkl"
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Could not load {path}: {e}")
        return None


def judge_form(bottom, start, spec, bundle):
    if bundle is not None:
        names = bundle["feature_names"]
        keys = [n.replace("_bottom", "") for n in names]
        x = np.array([[bottom[k] - start[k] for k in keys]], dtype=np.float32)
        probs = bundle["pipeline"].predict_proba(x)[0]
        idx = int(np.argmax(probs))
        label = bundle["label_encoder"].inverse_transform([idx])[0]
        return label == "good", f"{label} ({probs[idx]:.0%})"

    problems = [msg for feat, op, val, msg in spec["rules"]
                if not OPS[op](bottom[feat], val)]
    return (not problems), ("good" if not problems else "; ".join(problems))


# -- Recording ---------------------------------------------------------------
def append_row(exercise, label, bottom, start):
    os.makedirs("data", exist_ok=True)
    path = f"data/{exercise}.csv"
    deltas = {k: round(bottom[k] - start[k], 2) for k in bottom}
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["exercise", "label"] + [f"{k}_bottom" for k in deltas])
        writer.writerow([exercise, label] + list(deltas.values()))


# -- Drawing -----------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, WHITE, YELLOW = (0, 200, 0), (0, 0, 255), (255, 255, 255), (0, 220, 255)


def draw_skeleton(frame, lm, color):
    h, w = frame.shape[:2]
    for a, b in CONNECTIONS:
        if (lm[a].visibility or 0) < 0.4 or (lm[b].visibility or 0) < 0.4:
            continue
        cv2.line(frame, (int(lm[a].x * w), int(lm[a].y * h)),
                 (int(lm[b].x * w), int(lm[b].y * h)), color, 3)
    for p in lm:
        if (p.visibility or 0) > 0.4:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, WHITE, -1)


def draw_panel(frame, lines):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (430, 30 + 28 * len(lines)), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (12, 28 + 28 * i), FONT, 0.62, color, 2, cv2.LINE_AA)
    return frame


# -- Main --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", default="squat", choices=EXERCISES.keys())
    parser.add_argument("--source", default="0", help="0 for webcam or a video path")
    parser.add_argument("--record", action="store_true", help="save reps to data/<exercise>.csv")
    args = parser.parse_args()

    spec = EXERCISES[args.exercise]
    driver = spec["driver"]
    bundle = None if args.record else load_form_model(args.exercise)
    landmarker = build_landmarker()

    source = int(args.source) if args.source.isdigit() else args.source
    is_file = isinstance(source, str)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Could not open the video source. Try --source 1 for another camera.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    mode = "SVM model" if bundle else "angle rules"
    print(f"Exercise: {args.exercise} | form check: {mode} | record: {args.record}")

    t0, frame_i, last_ts = time.time(), 0, -1
    side, cal_samples, start, counter = None, [], None, None
    reps, log = 0, []
    last = None
    current_label = "good"

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        t = frame_i / fps if is_file else time.time() - t0
        ts = max(int(t * 1000), last_ts + 1)
        last_ts, frame_i = ts, frame_i + 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(Image(image_format=ImageFormat.SRGB, data=rgb), ts)

        lines = []
        skeleton_color = (0, 180, 255)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            if counter is None:
                if side is None:
                    side = choose_side(lm)
                if joints_visible(lm, side, spec):
                    cal_samples.append(compute_features(lm, side, spec, w, h))
                    lines.append((f"Hold still... {len(cal_samples)}/{CAL_FRAMES}", YELLOW))
                else:
                    side, cal_samples = None, []
                    lines.append(("Show your whole side to the camera", RED))
                lines.append((spec["hint"], WHITE))

                if len(cal_samples) >= CAL_FRAMES:
                    start = {k: float(np.median([s[k] for s in cal_samples])) for k in cal_samples[0]}
                    counter = RepCounter(start[driver])
                    print(f"Calibrated ({side} side). Start {driver} angle: {start[driver]:.0f}")

            else:
                if joints_visible(lm, side, spec):
                    feats = compute_features(lm, side, spec, w, h)
                    angle, event = counter.update(feats[driver], feats, t)

                    if event and event.get("partial"):
                        last = {"ok": False, "msg": "Partial rep, not counted", "t": t, "rom": event["rom"]}
                    elif event:
                        reps += 1
                        good, msg = judge_form(event["bottom"], start, spec, bundle)
                        note = "Descent too fast" if event["down"] < MIN_DOWN_TIME else ""
                        last = {"ok": good, "msg": msg, "t": t, "rom": event["rom"],
                                "down": event["down"], "up": event["up"], "note": note}
                        log.append(last)
                        if args.record:
                            append_row(args.exercise, current_label, event["bottom"], start)
                else:
                    lines.append(("Keep your whole side visible", RED))

                lines.insert(0, (f"{args.exercise.upper()}   Reps: {reps}", WHITE))
                lines.append((f"{driver} angle: {counter.smoothed:.0f}   ({counter.state})", WHITE))

                if args.record:
                    lines.append((f"RECORDING label: {current_label}  (keys 1/2/3)", YELLOW))

                if last:
                    color = GREEN if last["ok"] else RED
                    lines.append((f"Form: {last['msg']}", color))
                    if "down" in last:
                        total = last["down"] + last["up"]
                        lines.append((f"ROM {last['rom']:.0f} deg | down {last['down']:.1f}s | "
                                      f"up {last['up']:.1f}s | rep {total:.1f}s", WHITE))
                        if last["note"]:
                            lines.append((last["note"], YELLOW))
                    if t - last["t"] < 1.5:
                        skeleton_color = color

            draw_skeleton(frame, lm, skeleton_color)
        else:
            lines.append(("No person detected", RED))

        frame = draw_panel(frame, lines)
        cv2.imshow("Form coach", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            side, cal_samples, start, counter = None, [], None, None
        if args.record and key in LABEL_KEYS:
            current_label = LABEL_KEYS[key]

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    print("\n--- Session summary ---")
    print(f"Reps counted: {reps}")
    if log:
        counted = [r for r in log if "down" in r]
        good = sum(1 for r in counted if r["ok"])
        print(f"Good form: {good}/{len(counted)}")
        if counted:
            print(f"Avg ROM: {np.mean([r['rom'] for r in counted]):.0f} deg")
            print(f"Avg down: {np.mean([r['down'] for r in counted]):.1f}s | "
                  f"Avg up: {np.mean([r['up'] for r in counted]):.1f}s")


if __name__ == "__main__":
    main()
