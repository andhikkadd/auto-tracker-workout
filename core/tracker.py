import cv2
import numpy as np
import mediapipe as mp
import winsound
import math
from types import SimpleNamespace
import threading
# import time

from core.config import STATE
from core.ui import render_ui

mp_pose = mp.solutions.pose

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

def handle_mode_menu(key):
    if key == ord('7'):
        return "Bicep Curl"
    if key == ord('8'):
        return "Pull Up"
    if key == ord('9'):
        return "Squat"
    
    return None

def set_active_mode(new_mode):
    if new_mode is None:
        return 
    
    if new_mode == STATE["active"]:
        return 
    
    curl = STATE["curl"]
    pull = STATE["pull"]
    squat = STATE["squat"]  
        
    curl["L"]["reps"] = curl["R"]["reps"] = curl["prev_total"] = 0
    pull["reps"] = pull["prev_total"] = 0
    squat["reps"] = squat["prev_total"] = 0
    curl["L"]["stage"] = curl["R"]["stage"] = pull["stage"] = squat["stage"] = None
    
    curl["L"]["down_hold"] = curl["L"]["up_hold"] = 0
    curl["R"]["down_hold"] = curl["R"]["up_hold"] = 0
    pull["up_hold"] = pull["down_hold"] = 0
    squat["up_hold"] = squat["down_hold"] = 0
    
    STATE["active"] = new_mode
    beep_async(700, 80)

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

    step_side("L", features["ang_curl_L"], features["l_shoulder"], features["l_elbow"], features["l_wrist"])
    step_side("R", features["ang_curl_R"], features["r_shoulder"], features["r_elbow"], features["r_wrist"])

    after_total = curl["L"]["reps"] + curl["R"]["reps"]
    did_rep = after_total > curl["prev_total"]
    if did_rep:
        curl["prev_total"] = after_total
    return did_rep

def process_pull(features):
    pull = STATE["pull"]
    
    hands_overhead = features.get("hands_overhead", False)
    hold_n = pull["cfg"]["hold_n"]
    down_th = 140
    up_th = 70
    
    elbow_min = min(features["ang_curl_L"], features["ang_curl_R"])
    
    if not hands_overhead:
        pull["stage"] = None
        pull["down_hold"] = 0
        pull["up_hold"] = 0
        return False
    
    if elbow_min > down_th:
        label = "down"
    elif elbow_min < up_th:
        label = "up"
    else:
        label = "mid"
        
    if label == "down":
        pull["down_hold"] += 1
    else:
        pull["down_hold"] = 0
        
    if label == "up":
        pull["up_hold"] += 1
    else:
        pull["up_hold"] = 0

    if pull["stage"] is None:
        if label == "down" and pull["down_hold"] >= hold_n:
            pull["stage"] = "down"
        return False
    
    if pull["stage"] == "down" and label == "up" and pull["up_hold"] >= (hold_n - 1):
        pull["reps"] += 1
        pull["stage"] = "up"
        return True
    
    if pull["stage"] == "up" and label == "down" and pull["down_hold"] >= hold_n:
        pull["stage"] = "down"

def process_squat(features):
    squat = STATE["squat"]
    hold_n = squat["cfg"]["hold_n"]
    v = STATE["v"]
    
    down_th = 105
    up_th = 160
    
    ang_squ_L = features["ang_squat_L"]
    ang_squ_R = features["ang_squat_R"]
    knee_ang = min(ang_squ_L, ang_squ_R)
    
    if knee_ang < down_th:
        label = "squat down"
    elif knee_ang > up_th:
        label = "squat up"
    else:
        label = "mid"
        
    if label == "squat down":
        squat["stage"] = "down"
        
    if label == "squat up" and squat["stage"] == "down":
        squat["reps"] += 1
        squat["stage"] = "up"
        
    did_rep = squat["reps"] > squat["prev_total"]
    if did_rep:
        squat["prev_total"] = squat["reps"]
    return did_rep
    

def detect_pose(frame, key=None):
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results_pose = pose.process(rgb)

    if not results_pose.pose_landmarks:
        return frame

    lm = results_pose.pose_landmarks.landmark
    features = extract_features(lm, frame.shape[:2])
    
    new_mode = handle_mode_menu(key)
    set_active_mode(new_mode)
    
    if STATE.get("pending_mode") is not None:
        set_active_mode(STATE["pending_mode"])
        STATE["pending_mode"] = None


    update_active_counter(features)

    frame = render_ui(frame, features, beep_async)
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

    ang_curl_L = angle((l_shoulder.x, l_shoulder.y),
                (l_elbow.x, l_elbow.y),
                (l_wrist.x, l_wrist.y))
    
    ang_curl_R = angle((r_shoulder.x, r_shoulder.y),
                (r_elbow.x, r_elbow.y),
                (r_wrist.x, r_wrist.y))
    
    # curl
    elbow_bent = min(ang_curl_L, ang_curl_R) < 120

    # pull
    # pull_delta = ((l_shoulder.y + r_shoulder.y) / 2) - ((l_wrist.y + r_wrist.y) / 2)
    # pull_elbow_straight = min(ang_curl_L, ang_curl_R) > 150
    
    hands_overhead = (
        l_wrist.visibility > STATE["v"] and r_wrist.visibility > STATE["v"] and
        l_elbow.visibility > STATE["v"] and r_elbow.visibility > STATE["v"] and
        l_shoulder.visibility > STATE["v"] and r_shoulder.visibility > STATE["v"] and
        l_wrist.y < l_shoulder.y and r_wrist.y < r_shoulder.y)

    # squat
    ang_squat_L = angle((l_hip.x, l_hip.y),
                        (l_knee.x, l_knee.y),
                        (l_ankle.x, l_ankle.y))
    
    ang_squat_R = angle((r_hip.x, r_hip.y),
                        (r_knee.x, r_knee.y),
                        (r_ankle.x, r_ankle.y))
    
    knee_bent = min(ang_squat_L, ang_squat_R) < 150
    
    vis_lower_L = (l_hip.visibility > STATE["v"] and l_knee.visibility > STATE["v"] and l_ankle.visibility > STATE["v"])
    vis_lower_R = (r_hip.visibility > STATE["v"] and r_knee.visibility > STATE["v"] and r_ankle.visibility > STATE["v"])
    vis_ok_lower = vis_lower_L or vis_lower_R

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
        
        # curl
        "ang_curl_L": ang_curl_L,
        "ang_curl_R": ang_curl_R,
        "elbow_bent": elbow_bent,
        
        # pull
        # "pull_delta": pull_delta,
        "hands_overhead": hands_overhead,
        
        "vis_ok_upper": (
            l_shoulder.visibility > STATE["v"] and r_shoulder.visibility > STATE["v"] and
            l_elbow.visibility > STATE["v"] and r_elbow.visibility > STATE["v"]),
        
        # "pull_ready": (hands_overhead),
        
        # squat
        "ang_squat_L": ang_squat_L,
        "ang_squat_R": ang_squat_R,
        
        "vis_ok_lower": vis_ok_lower,
        "knee_bent" : knee_bent,
    }

def update_active_counter(features):
    if STATE["active"] == "Bicep Curl":
        if process_curl(features):
            beep_async(3000, 120)
    if STATE["active"] == "Pull Up":
        if process_pull(features):
            beep_async(3000, 120)
    if STATE["active"] == "Squat":
        if process_squat(features):
            beep_async(3000, 120)