import cv2
from core.config import STATE

def inside(px, py, x, y, w, h):
    return x <= px <= x + w and y <= py <= y + h

def render_ui(frame, features, beep_async):
    btn = STATE["BTN"]
    btc = STATE["BTC"]
    curl = STATE["curl"]
    pull = STATE["pull"]
    squat = STATE["squat"]
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
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1
    )

    # INFO BOX
    hud_x, hud_y = 520, 10
    hud_w, hud_h = 220, 70

    cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h + 10), (0,0,0), -1)
    cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h + 10), (255,255,255), 1)

    if STATE["active"] is None:
        cv2.putText(frame, "SELECT", (hud_x + 28, hud_y + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, "MODE", (hud_x + 33, hud_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    else:
        if STATE["active"] == "Bicep Curl":
            cv2.putText(frame, STATE["active"], (hud_x + 10, hud_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            total = curl["L"]["reps"] + curl["R"]["reps"]

            cv2.putText(frame, f"L: {curl['L']['reps']} | R: {curl['R']['reps']}",
                        (hud_x + 10, hud_y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            cv2.putText(frame, f"Total: {total}",
                        (hud_x + 10, hud_y + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

        elif STATE["active"] == "Pull Up":
            cv2.putText(frame, STATE["active"], (hud_x + 25, hud_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            total = pull["reps"]
            cv2.putText(frame, f"Total: {total}",
                        (hud_x + 25, hud_y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
        
        elif STATE["active"] == "Squat":
            cv2.putText(frame, STATE["active"], (hud_x + 25, hud_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            total = squat["reps"]
            cv2.putText(frame, f"Total: {total}",
                        (hud_x + 25, hud_y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
            
    for b in STATE["btns"]:
        active = (STATE["active"] == b["mode"])
        fill = (80,80,80) if active else (50,50,50)

        cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), fill, -1)
        cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (255,255,255), 1)

        cv2.putText(frame, b["label"], (b["x1"]+12, b["y1"]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    def hit_mode_button(px, py):
        for b in STATE["btns"]:
            if b["x1"] <= px <= b["x2"] and b["y1"] <= py <= b["y2"]:
                return b
        return None

    mode_touch = None

    if features["l_index"].visibility > STATE["v"]:
        x, y = features["li_px"]
        mode_touch = hit_mode_button(x, y) or mode_touch

    if features["r_index"].visibility > STATE["v"]:
        x, y = features["ri_px"]
        mode_touch = hit_mode_button(x, y) or mode_touch

    # arm/hold mirip reset
    if mode_touch is None:
        STATE["mode_armed"] = True
        STATE["mode_hold"] = 0
        STATE["mode_target"] = None
    else:
        # kalau pindah tombol saat nahan, reset hold
        if STATE["mode_target"] != mode_touch["id"]:
            STATE["mode_target"] = mode_touch["id"]
            STATE["mode_hold"] = 0

        if STATE["mode_armed"]:
            STATE["mode_hold"] += 1

        if STATE["mode_hold"] >= STATE["mode_hold_n"] and STATE["mode_armed"]:
            STATE["pending_mode"] = mode_touch["mode"]
    
            STATE["mode_armed"] = False
            STATE["mode_hold"] = STATE["mode_hold_n"]

        

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
        squat["reps"] = squat["prev_total"] = 0
        curl["L"]["stage"] = curl["R"]["stage"] = pull["stage"] = squat["stage"] = None
        curl["L"]["down_ref"] = curl["L"]["up_ref"] = None
        curl["R"]["down_ref"] = curl["R"]["up_ref"] = None
        pull["top_ref"] = pull["bottom_ref"] = None
        squat["up_ref"] = squat["down_ref"] = None
        curl["L"]["down_hold"] = curl["L"]["up_hold"] = 0
        curl["R"]["down_hold"] = curl["R"]["up_hold"] = 0
        pull["up_hold"] = pull["down_hold"] = 0
        squat["up_hold"] = squat["down_hold"] = 0

        beep_async(700, 200)

        STATE["reset_armed"] = False
        STATE["reset_hold"] = STATE["reset_hold_n"]

    # RESET PROGRESS BAR
    overlay = frame.copy()

    cv2.rectangle(overlay, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (80,80,80), -1)
    cv2.rectangle(overlay, (bar_x1, bar_y), (bar_x1 + bar_w, bar_y + bar_h), (0,255,0), -1)

    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # LANDMARKS
    cv2.circle(frame, features["li_px"], 6, (255,255,255), -1)
    cv2.circle(frame, features["ri_px"], 6, (255,255,255), -1)

    # LANDMARK LINES
    H, W = frame.shape[:2]
    
    mid_sh = (
        int((features["l_shoulder"].x + features["r_shoulder"].x) * 0.5 * W),
        int((features["l_shoulder"].y + features["r_shoulder"].y) * 0.5 * H)
    )

    mid_hp = (
        int((features["l_hip"].x + features["r_hip"].x) * 0.5 * W),
        int((features["l_hip"].y + features["r_hip"].y) * 0.5 * H)
    )

    # badan 1 garis (stickman)
    cv2.line(frame, mid_sh, mid_hp, (255,255,255), 3)


    def p(lm):
        return int(lm.x * W), int(lm.y * H)
    
    
    # pull_stage = pull["stage"]
    # pull_hold = pull["down_hold"]
    # hands_over = features["hands_overhead"]
    # elbow_min = min(features["ang_curl_L"], features["ang_curl_R"])
    
    
    # cv2.putText(
    #     frame, f"stage: {pull_stage}", (20, 80),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2
    # )
    # cv2.putText(
    #     frame, f"angle: {elbow_min:.3f}", (20, 100),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2
    # )
    # cv2.putText(
    #     frame, f"hand over: {hands_over}", (20, 140),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2
    # )
    # cv2.putText(
    #     frame, f"pull hold: {pull_hold}", (20, 160),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2
    # )

    cv2.line(frame, p(features["l_shoulder"]), p(features["l_elbow"]), (255,255,255), 3)
    cv2.line(frame, p(features["l_elbow"]), p(features["l_index"]), (255,255,255), 3)
    cv2.line(frame, p(features["r_shoulder"]), p(features["r_elbow"]), (255,255,255), 3)
    cv2.line(frame, p(features["r_elbow"]), p(features["r_index"]), (255,255,255), 3)
    cv2.line(frame, p(features["l_shoulder"]), p(features["r_shoulder"]), (255,255,255), 3)

    cv2.line(frame, p(features["l_hip"]), p(features["l_knee"]), (255,255,255), 3)
    cv2.line(frame, p(features["l_knee"]), p(features["l_ankle"]), (255,255,255), 3)
    cv2.line(frame, p(features["r_hip"]), p(features["r_knee"]), (255,255,255), 3)
    cv2.line(frame, p(features["r_knee"]), p(features["r_ankle"]), (255,255,255), 3)

    cv2.line(frame, p(features["l_shoulder"]), p(features["r_shoulder"]), (255,255,255), 3)
    cv2.line(frame, p(features["l_hip"]), p(features["r_hip"]), (255,255,255), 3)
    # cv2.line(frame, p(features["l_index"]), p(features["r_index"]), (0,0,255), 2)

    return frame