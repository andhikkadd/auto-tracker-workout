import cv2

def inside(px, py, x, y, w, h):
    return x <= px <= x + w and y <= py <= y + h

def render_ui(frame, features, STATE, beep_async):
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

        cv2.putText(frame, f"L: {curl['L']['reps']} | R: {curl['R']['reps']}",
                    (hud_x + 10, hud_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.putText(frame, f"Total: {total}",
                    (hud_x + 10, hud_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

    elif STATE["active"] == "Pull Up":
        total = pull["reps"]
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

    def p(lm):
        return int(lm.x * W), int(lm.y * H)

    cv2.line(frame, p(features["l_shoulder"]), p(features["l_elbow"]), (255,255,255), 3)
    cv2.line(frame, p(features["l_elbow"]), p(features["l_index"]), (255,255,255), 3)
    cv2.line(frame, p(features["r_shoulder"]), p(features["r_elbow"]), (255,255,255), 3)
    cv2.line(frame, p(features["r_elbow"]), p(features["r_index"]), (255,255,255), 3)

    cv2.line(frame, p(features["l_hip"]), p(features["l_knee"]), (255,255,255), 3)
    cv2.line(frame, p(features["l_knee"]), p(features["l_ankle"]), (255,255,255), 3)

    cv2.line(frame, p(features["r_hip"]), p(features["r_knee"]), (255,255,255), 3)
    cv2.line(frame, p(features["r_knee"]), p(features["r_ankle"]), (255,255,255), 3)

    return frame