import cv2
import numpy as np
import mediapipe as mp
import winsound
import time
import math
from types import SimpleNamespace
import threading

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
 
STATE = {
    "BTN": {"x": 20, "y": 20, "w": 80, "h": 40},
    "HUD": {"x": 520, "y": 10, "w": 220, "h": 70},
    "FPS_POS": (10, 470),
    
    "curl" : {
        "prev_total" : 0,
        
        "cfg" : {
            "reset_hold_n" : 8, 
            "hold_n" : 4,
            "margin": 10,   
        },
        
        "L" : {
            "reps" : 0,
            "stage" : None,
            "down_ref" : None,
            "up_ref" : None,
            "down_hold" : 0,
            "up_hold" : 0,
        },
        "R" : {
            "reps" : 0,
            "stage" : None,
            "down_ref" : None,
            "up_ref" : None,
            "down_hold" : 0,
            "up_hold" : 0,
        },
    },    
    
    "pull" : {
        "prev_total" : 0,
        
        "cfg" : {
            "hold_n" : 2, 
            "margin_y": 0.03,
        },
        
        "reps" : 0,
        "stage" : None,   
        "top_ref" : None,        
        "bottom_ref" : None,     
        "top_hold" : 0,
        "bottom_hold" : 0,
        
    },
    
    "reset_hold": 0,
    "reset_armed": True,
    "reset_hold_n": 8,
    
    "smooth_lm" : {},
    "smooth_alpha" : 0.6,  
    "smooth_max_step": 0.15,
    
    "v": 0.5,         
    "alpha": 0.25,   
    "active_hold_n": 4,   
    
    "cand_curl" : 0,
    "cand_pull" : 0,
    "active" : None,
 

    "prev_t": time.time(),
    "fps": 0.0,
}

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

        # UPDATE HOLDS UTK STABOLISASI REF DOWN / UP
        if S["down_ref"] is not None and ang < S["down_ref"] - MARGIN:
            S["down_hold"] += 1
        else:
            S["down_hold"] = 0

        if S["up_ref"] is not None and ang < S["up_ref"] - MARGIN:
            S["up_hold"] += 1
        else:
            S["up_hold"] = 0

        # UPDATE REF JIKA STABIL BEBERAPA FRAME
        if S["down_hold"] >= HOLD_N:
            S["down_ref"] = ang if S["down_ref"] is None else (1 - ALPHA) * S["down_ref"] + ALPHA * ang

        if S["up_hold"] >= HOLD_N:
            S["up_ref"] = ang if S["up_ref"] is None else (1 - ALPHA) * S["up_ref"] + ALPHA * ang

        # DEFAULT (FALLBACK JIKA REF BELUM TERBENTUK)
        a_high = 150  
        a_low  = 50   

        if S["down_ref"] is not None:
            a_high = S["down_ref"] - MARGIN
        if S["up_ref"] is not None:
            a_low = S["up_ref"] + MARGIN

        # KLASIFIKASI POSISI DOWN/MID/UP
        label = "NOT_CURL"
        if elbow.visibility > v and shoulder.visibility > v and wrist.visibility > v:
            if ang < a_low:
                label = "curl up"
            elif ang > a_high:
                label = "curl down"
            else:
                label = "mid"

        # STAGE + REPS
        if label == "curl down":
            S["stage"] = "down"

        if label == "curl up" and S["stage"] == "down":
            S["reps"] += 1
            S["stage"] = "up"
            winsound.Beep(3000, 120)

        # DEBUG
        S["label"] = label

    step_side("L", features["ang_L"], features["l_shoulder"], features["l_elbow"], features["l_wrist"])
    step_side("R", features["ang_R"], features["r_shoulder"], features["r_elbow"], features["r_wrist"])
    
    after_total = curl["L"]["reps"] + curl["R"]["reps"]
    did_rep = after_total > curl["prev_total"]
    return did_rep

def process_pull(features ):
    lw = features["l_wrist"]
    rw = features["r_wrist"]
    ls = features["l_shoulder"]
    rs = features["r_shoulder"]

    wrist_y = (lw.y + rw.y) / 2
    shoulder_y = (ls.y + rs.y) / 2

    delta = shoulder_y - wrist_y 

    # HOLD = STATE["pull"]["cfg"]["hold_n"]
    # ALPHA = STATE["alpha"]
    # MARGIN = STATE["pull"]["cfg"]["margin_y"]
    pull = STATE["pull"]

    # DETECT BOTTOM 
    if pull["bottom_ref"] is None or delta > pull["bottom_ref"] - pull["cfg"]["margin_y"]:
        pull["bottom_hold"] += 1
    else:
        pull["bottom_hold"] = 0

    if pull["bottom_hold"] >= pull[]:
        pull["bottom_ref"] = (
            delta if pull["bottom_ref"] is None
            else pull["bottom_ref"] * (1 - ALPHA) + delta * ALPHA
        )
        pull["stage"] = "down"

    # === detect top ===
    if pull["top_ref"] is None or delta < pull["top_ref"] + MARGIN:
        pull["top_hold"] += 1
    else:
        pull["top_hold"] = 0

    if pull["top_hold"] >= HOLD:
        pull["top_ref"] = (
            delta if pull["top_ref"] is None
            else pull["top_ref"] * (1 - ALPHA) + delta * ALPHA
        )

        if pull["stage"] == "down":
            pull["reps"] += 1
            pull["stage"] = "up"
            
    return pu

def inside(px, py, x, y, w, h):
    return x <= px <= x + w and y <= py <= y + h

def detect_pose(frame):
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks:
        return frame
    
    lm = results.pose_landmarks.landmark
    features = extract_features(lm, frame.shape[:2])
    
    STATE["active"] = detect_exercise(features)

    update_active_counter(features)

    frame = render_ui(frame, features)
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
    r_shoulder = smooth_landmark("r_shoulder", lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], STATE["smooth_alpha"]) 
    r_elbow = smooth_landmark("r_elbow", lm[mp_pose.PoseLandmark.RIGHT_ELBOW], STATE["smooth_alpha"]) 
    r_wrist = smooth_landmark("r_wrist", lm[mp_pose.PoseLandmark.RIGHT_WRIST], STATE["smooth_alpha"]) 
    r_index = smooth_landmark("r_index", lm[mp_pose.PoseLandmark.RIGHT_INDEX], STATE["smooth_alpha"]) 
    
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
        # before = STATE["curl"]["L"]["reps"] + STATE["curl"]["R"]["reps"]y
        process_curl(features)
        # after = STATE["curl"]["L"]["reps"] + STATE["curl"]["R"]["reps"]

        # if after > STATE["curl"]["prev_total"]:
        #     STATE["curl"]["prev_total"] = after
            
    if STATE["active"] == "Pull Up":
        # before = STATE["pull"]["reps"]
        process_pull(features)
    return 

def render_ui(frame, features):
    btn = STATE["BTN"]
    curl = STATE["curl"]
    pull = STATE["pull"]

    # UI CONFIG
    bar_h = 4
    pad = 5
    bar_y = btn["y"] + btn["h"] - bar_h - pad

    den = max(STATE["reset_hold_n"], 1)
    progress = min(STATE["reset_hold"] / den, 1.0)

    bar_x1 = btn["x"] + pad
    bar_x2 = btn["x"] + btn["w"] - pad
    bar_w = int((bar_x2 - bar_x1) * progress)

    # FPS
    cv2.putText(
        frame, f"FPS: {STATE['fps']:.1f}", (10, 470),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1
    )

    # INFO BOX
    hud_x, hud_y = 520, 10
    hud_w, hud_h = 220, 70

    cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h + 10), (0,0,0), -1)
    cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h + 10), (255,255,255), 1)

    title = "" if STATE["active"] is None else STATE["active"]
    cv2.putText(frame, title, (hud_x + 10, hud_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    if STATE["active"] == "Bicep Curl":
        total = curl["L"]["reps"] + curl["R"]["reps"]

        if total > curl["prev_total"]:
            curl["prev_total"] = total

        cv2.putText(frame, f"L: {curl['L']['reps']} | R: {curl['R']['reps']}",
                    (hud_x + 10, hud_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.putText(frame, f"Total: {total}",
                    (hud_x + 10, hud_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

    elif STATE["active"] == "Pull Up":
        total = pull["reps"]

        if total > pull["prev_total"]:
            pull["prev_total"] = total

        cv2.putText(frame, f"Total: {total}",
                    (hud_x + 10, hud_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

    # RESET BUTTON
    cv2.rectangle(frame,
        (btn["x"], btn["y"]),
        (btn["x"] + btn["w"], btn["y"] + btn["h"]),
        (0,0,0), -1
    )
    cv2.rectangle(frame,
        (btn["x"], btn["y"]),
        (btn["x"] + btn["w"], btn["y"] + btn["h"]),
        (255,255,255), 1
    )
    cv2.putText(frame, "RESET",
        (btn["x"] + 15, btn["y"] + 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1
    )

    # TOUCH DETECTION
    touch = False

    if features["l_index"].visibility > STATE["v"]:
        x, y = features["li_px"]
        touch |= inside(x, y, btn["x"], btn["y"], btn["w"], btn["h"])

    if features["r_index"].visibility > STATE["v"]:
        x, y = features["ri_px"]
        touch |= inside(x, y, btn["x"], btn["y"], btn["w"], btn["h"])

    if not touch:
        STATE["reset_armed"] = True
        STATE["reset_hold"] = 0

    if touch and STATE["reset_armed"]:
        STATE["reset_hold"] += 1

    if STATE["reset_hold"] >= STATE["reset_hold_n"] and STATE["reset_armed"]:
        curl["L"]["reps"] = curl["R"]["reps"] = curl["prev_total"] = 0
        pull["reps"] = pull["prev_total"] = 0

        winsound.Beep(700, 200)

        STATE["reset_armed"] = False
        STATE["reset_hold"] = STATE["reset_hold_n"]

    # RESET PROGRESS BAR 
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (bar_x1, bar_y),
        (bar_x2, bar_y + bar_h),
        (80,80,80),
        -1
    )

    cv2.rectangle(
        overlay,
        (bar_x1, bar_y),
        (bar_x1 + bar_w, bar_y + bar_h),
        (0,255,0),
        -1
    )

    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # LANDMARKS 
    cv2.circle(frame, features["li_px"], 6, (255,255,255), -1)
    cv2.circle(frame, features["ri_px"], 6, (255,255,255), -1)

    # LANDMARK LINES
    H, W = frame.shape[:2]

    def p(lm):
        return int(lm.x * W), int(lm.y * H)

    cv2.line(frame, p(features["l_shoulder"]), p(features["l_elbow"]), (255,255,255), 3)
    cv2.line(frame, p(features["l_elbow"]), p(features["l_index"]), (255,255,255), 3)

    cv2.line(frame, p(features["r_shoulder"]), p(features["r_elbow"]), (255,255,255), 3)
    cv2.line(frame, p(features["r_elbow"]), p(features["r_index"]), (255,255,255), 3)

    return frame


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("tidak dpt membuka kamera")
    exit()

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print("gagal menangkap frame")
        break 
    
    now = time.time()
    dt = now - STATE["prev_t"]
    STATE["prev_t"] = now
    if dt > 0:
        inst_fps = 1.0 / dt
        STATE["fps"] = STATE["fps"] * 0.9 + inst_fps * 0.1

    frame = detect_pose(frame)
    
    cv2.imshow("Detect",frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
