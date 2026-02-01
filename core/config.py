import time

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

    "squat" : {
        "prev_total" : 0
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