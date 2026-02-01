import cv2
import numpy as np
import mediapipe as mp
import winsound
import time
import math
from types import SimpleNamespace
import threading

from config import STATE
from ui import render_ui

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def beep_async(freq=3000, dur=120):
    threading.Thread(target=winsound.Beep, args=(freq, dur), daemon=True).start()

def angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    ang = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(ang)

def smooth_landmark(name, lm, alpha,):
    if name not in STATE["smooth_lm"]:
        STATE["smooth_lm"][name] = {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}

    prev = STATE["smooth_lm"][name]

    dx = lm.x - prev["x"]
    dy = lm.y - prev["y"]
    dist = math.hypot(dx, dy)

    if dist > STATE["smooth_max_step"]:
        s = STATE["smooth_max_step"] / dist
        lm_x = prev["x"] + dx * s
        lm_y = prev["y"] + dy * s
        lm_z = prev["z"] + (lm.z - prev["z"]) * s
        lm_v = prev["visibility"] + (lm.visibility - prev["visibility"]) * s
    else:
        lm_x, lm_y, lm_z, lm_v = lm.x, lm.y, lm.z, lm.visibility

    prev["x"] = prev["x"] * (1 - alpha) + lm_x * alpha
    prev["y"] = prev["y"] * (1 - alpha) + lm_y * alpha
    prev["z"] = prev["z"] * (1 - alpha) + lm_z * alpha
    prev["visibility"] = prev["visibility"] * (1 - alpha) + lm_v * alpha

    return SimpleNamespace(**prev)

def process_curl(features):
    curl = STATE["curl"]
    HOLD_N = curl["cfg"]["hold_n"]
    ALPHA = STATE["alpha"]
    MARGIN = curl["cfg"]["margin"]
    v = STATE["v"]

    def step_side(side_key, ang, shoulder, elbow, wrist):
        S = curl[side_key]

        if S["down_ref"] is not None and ang < S["down_ref"] - MARGIN:
            S["down_hold"] += 1
        else:
            S["down_hold"] = 0

        if S["up_ref"] is not None and ang < S["up_ref"] - MARGIN:
            S["up_hold"] += 1
        else:
            S["up_hold"] = 0

        if S["down_hold"] >= HOLD_N:
            S["down_ref"] = ang if S["down_ref"] is None else (1 - ALPHA) * S["down_ref"] + ALPHA * ang

        if S["up_hold"] >= HOLD_N:
            S["up_ref"] = ang if S["up_ref"] is None else (1 - ALPHA) * S["up_ref"] + ALPHA * ang

        a_high = 150
        a_low  = 50

        if S["down_ref"] is not None:
            a_high = S["down_ref"] - MARGIN
        if S["up_ref"] is not None:
            a_low = S["up_ref"] + MARGIN

        label = "NOT_CURL"
        if elbow.visibility > v and shoulder.visibility > v and wrist.visibility > v:
            if ang < a_low:
                label = "curl up"
            elif ang > a_high:
                label = "curl down"
            else:
                label = "mid"

        if label == "curl down":
            S["stage"] = "down"

        if label == "curl up" and S["stage"] == "down":
            S["reps"] += 1
            S["stage"] = "up"

        S["label"] = label

    step_side("L", features["ang_L"], features["l_shoulder"], features["l_elbow"], features["l_wrist"])
    step_side("R", features["ang_R"], features["r_shoulder"], features["r_elbow"], features["r_wrist"])

    after_total = curl["L"]["reps"] + curl["R"]["reps"]
    did_rep = after_total > curl["prev_total"]
    if did_rep:
        curl["prev_total"] = after_total
    return did_rep

def process_pull(features ):
    lw = features["l_wrist"]
    rw = features["r_wrist"]
    ls = features["l_shoulder"]
    rs = features["r_shoulder"]

    wrist_y = (lw.y + rw.y) / 2
    shoulder_y = (ls.y + rs.y) / 2

    delta = shoulder_y - wrist_y

    pull = STATE["pull"]

    if pull["bottom_ref"] is None or delta > pull["bottom_ref"] - pull["cfg"]["margin_y"]:
        pull["bottom_hold"] += 1
    else:
        pull["bottom_hold"] = 0

    if pull["bottom_hold"] >= pull["cfg"]["hold_n"]:
        pull["bottom_ref"] = (
            delta if pull["bottom_ref"] is None
            else pull["bottom_ref"] * (1 - STATE["alpha"]) + delta * STATE["alpha"]
        )
        pull["stage"] = "down"

    if pull["top_ref"] is None or delta < pull["top_ref"] + pull["cfg"]['margin_y']:
        pull["top_hold"] += 1
    else:
        pull["top_hold"] = 0

    if pull["top_hold"] >= pull["cfg"]["hold_n"]:
        pull["top_ref"] = (
            delta if pull["top_ref"] is None
            else pull["top_ref"] * (1 - STATE["alpha"]) + delta * STATE["alpha"]
        )

        if pull["stage"] == "down":
            pull["reps"] += 1
            pull["stage"] = "up"

    did_rep = pull["reps"] > pull["prev_total"]
    if did_rep:
        pull["prev_total"] = pull["reps"]
    return did_rep

def detect_pose(frame):
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks:
        return frame

    lm = results.pose_landmarks.landmark
    features = extract_features(lm, frame.shape[:2])

    STATE["active"] = detect_exercise(features)

    update_active_counter(features)

    # BEDANYA CUMA DI SINI: render_ui dikasih STATE dan beep_async
    frame = render_ui(frame, features, STATE, beep_async)
    return frame

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_features(lm, shape):
    H, W = shape

    l_shoulder = smooth_landmark("l_shoulder", lm[mp_pose.PoseLandmark.LEFT_SHOULDER], STATE["smooth_alpha"])
    l_elbow = smooth_landmark("l_elbow", lm[mp_pose.PoseLandmark.LEFT_ELBOW], STATE["smooth_alpha"])
    l_wrist = smooth_landmark("l_wrist", lm[mp_pose.PoseLandmark.LEFT_WRIST], STATE["smooth_alpha"])
    l_index = smooth_landmark("l_index", lm[mp_pose.PoseLandmark.LEFT_INDEX], STATE["smooth_alpha"])
    l_hip = smooth_landmark("l_hip", lm[mp_pose.PoseLandmark.LEFT_HIP], STATE["smooth_alpha"])
    l_knee = smooth_landmark("l_knee", lm[mp_pose.PoseLandmark.LEFT_KNEE], STATE["smooth_alpha"])
    l_ankle = smooth_landmark("l_ankle", lm[mp_pose.PoseLandmark.LEFT_ANKLE], STATE["smooth_alpha"])

    r_shoulder = smooth_landmark("r_shoulder", lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], STATE["smooth_alpha"])
    r_elbow = smooth_landmark("r_elbow", lm[mp_pose.PoseLandmark.RIGHT_ELBOW], STATE["smooth_alpha"])
    r_wrist = smooth_landmark("r_wrist", lm[mp_pose.PoseLandmark.RIGHT_WRIST], STATE["smooth_alpha"])
    r_index = smooth_landmark("r_index", lm[mp_pose.PoseLandmark.RIGHT_INDEX], STATE["smooth_alpha"])
    r_hip = smooth_landmark("r_hip", lm[mp_pose.PoseLandmark.RIGHT_HIP], STATE["smooth_alpha"])
    r_knee = smooth_landmark("r_knee", lm[mp_pose.PoseLandmark.RIGHT_KNEE], STATE["smooth_alpha"])
    r_ankle = smooth_landmark("r_ankle", lm[mp_pose.PoseLandmark.RIGHT_ANKLE], STATE["smooth_alpha"])

    ang_L = angle((l_shoulder.x, l_shoulder.y),
                (l_elbow.x, l_elbow.y),
                (l_wrist.x, l_wrist.y))

    ang_R = angle((r_shoulder.x, r_shoulder.y),
                (r_elbow.x, r_elbow.y),
                (r_wrist.x, r_wrist.y))

    hands_overhead = (
        l_wrist.visibility > STATE["v"] and r_wrist.visibility > STATE["v"] and
        l_shoulder.visibility > STATE["v"] and r_shoulder.visibility > STATE["v"] and
        l_wrist.y < l_shoulder.y and r_wrist.y < r_shoulder.y
    )

    return {
        "l_shoulder": l_shoulder, "r_shoulder": r_shoulder,
        "l_elbow": l_elbow, "r_elbow": r_elbow,
        "l_wrist": l_wrist, "r_wrist": r_wrist,
        "l_index": l_index, "r_index": r_index,
        "l_hip": l_hip, "r_hip": r_hip,
        "l_knee": l_knee, "r_knee": r_knee,
        "l_ankle": l_ankle, "r_ankle": r_ankle,

        "li_px" : (int(l_index.x * W), int(l_index.y * H)),
        "ri_px" : (int(r_index.x * W), int(r_index.y * H)),

        "ang_L": ang_L,
        "ang_R": ang_R,

        "hands_overhead": hands_overhead,

        "vis_ok_upper": (
            l_shoulder.visibility > STATE["v"] and r_shoulder.visibility > STATE["v"] and
            l_elbow.visibility > STATE["v"] and r_elbow.visibility > STATE["v"]
        )
    }

def detect_exercise(features):
    if not features["vis_ok_upper"]:
        STATE["cand_curl"] = 0
        STATE["cand_pull"] = 0
        return None

    if features["hands_overhead"]:
        STATE["cand_pull"] += 1
        STATE["cand_curl"] = 0
        if STATE["cand_pull"] >= STATE["active_hold_n"]:
            return "Pull Up"
    else:
        STATE["cand_curl"] += 1
        STATE["cand_pull"] = 0
        if STATE["cand_curl"] >= STATE["active_hold_n"]:
            return "Bicep Curl"

    return STATE["active"]

def update_active_counter(features):
    if STATE["active"] == "Bicep Curl":
        if process_curl(features):
            beep_async(3000, 120)

    if STATE["active"] == "Pull Up":
        if process_pull(features):
            beep_async(3000, 120)