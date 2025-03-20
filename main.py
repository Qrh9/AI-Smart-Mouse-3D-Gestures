import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import os
# ======================
# AI Smart Mouse - 3D Gestures
# ALL RIGHTS RESERVED © 2025
# Licensed under CC BY-NC-SA 4.0
#Developer: @allnught (Telegram)
# phone: +964 771 441 4110
# email: gl1qmyoj@gmail.com
#License: https://creativecommons.org/licenses/by-nc-sa/4.0/
# ======================

# this is a first version of the AI Smart Mouse - 3D Gestures, it's a simple version, and it's not perfect, but it's a good start
# if you want to improve it, you can fork it and edit it as you want, but please keep the credits and the license.
# if you have any questions, you can contact me on Telegram: @allnught
pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)  # 60 FPS for smoother experience 

screen_width, screen_height = pyautogui.size()

prev_x, prev_y, prev_z = None, None, None
mouse_x, mouse_y = screen_width // 2, screen_height // 2

xy_sensitivity = 2.5   
z_sensitivity = 350.0
pinch_click_threshold = 45 

click_triggered = False
# ======================
# (Dynamic Moves)
# ======================
alpha = 0.85
movement_deadzone = 2.0  

def finger_ext(landmarks, tip_idx, pip_idx):
    return landmarks.landmark[tip_idx].y < landmarks.landmark[pip_idx].y

def dynamicmoves(old_val, new_val, speed, base_alpha=0.85):
    if old_val is None:
        return new_val
    alpha_dynamic = max(0.5, base_alpha - speed * 0.02)
    return alpha_dynamic * old_val + (1 - alpha_dynamic) * new_val

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    movement_mode = False
    nstop = False
    scroll_mode = False
    reset_mode = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            thumb_tip = hand_landmarks.landmark[4]

            ix, iy = index_tip.x * w, index_tip.y * h
            iz = index_tip.z

            index_extended = finger_ext(hand_landmarks, 8, 6)
            middle_extended = finger_ext(hand_landmarks, 12, 10)
            thumb_extended = finger_ext(hand_landmarks, 4, 2)

            if index_extended and middle_extended:
                movement_mode = True
                nstop = False
            elif index_extended and not middle_extended:
                nstop = True
                movement_mode = False
                prev_x, prev_y, prev_z = None, None, None

            if not index_extended and not middle_extended and not thumb_extended:
                scroll_mode = True
            else:
                scroll_mode = False

            snap_distance = np.hypot((thumb_tip.x * w - middle_tip.x * w), 
                                   (thumb_tip.y * h - middle_tip.y * h))
            
            snap_threshold = 25
            snap_release_threshold = 35
            
            if snap_distance < snap_threshold and not reset_mode:
                reset_mode = True
                mouse_x, mouse_y = screen_width // 2, screen_height // 2
            elif snap_distance > snap_release_threshold:
                reset_mode = False

            # ======================
            # (Move) - Mouse Control
            # ======================
            if movement_mode:
                if prev_x is None:
                    prev_x, prev_y, prev_z = ix, iy, iz

                speed_x = abs(ix - prev_x)
                speed_y = abs(iy - prev_y)
                speed_z = abs(iz - prev_z)
                speed = (speed_x + speed_y + speed_z) / 3.0

                ix = dynamicmoves(prev_x, ix, speed, alpha)
                iy = dynamicmoves(prev_y, iy, speed, alpha)
                iz = dynamicmoves(prev_z, iz, speed, alpha)

                dx = ix - prev_x
                dy = iy - prev_y
                dz = iz - prev_z

                if abs(dx) < movement_deadzone:
                    dx = 0
                if abs(dy) < movement_deadzone:
                    dy = 0

                dx_scaled = (dx / w) * screen_width * xy_sensitivity
                dy_scaled = (dy / h) * screen_height * xy_sensitivity

                mouse_x += dx_scaled
                mouse_y += dy_scaled
                mouse_y -= dz * z_sensitivity  # Invert the Z-axis

                mouse_x = max(0, min(screen_width, mouse_x))
                mouse_y = max(0, min(screen_height, mouse_y))

                pyautogui.moveTo(mouse_x, mouse_y, duration=0)

                prev_x, prev_y, prev_z = ix, iy, iz
                click_triggered = False  # Reset click trigger

            # ======================
            # (Click)
            # ======================
            elif nstop:
                prev_x, prev_y, prev_z = None, None, None
                distance_click = np.hypot((ix - thumb_tip.x * w), (iy - thumb_tip.y * h))
                
                mouse_x, mouse_y = pyautogui.position()
                pyautogui.moveTo(mouse_x, mouse_y, duration=0)
                
                if distance_click < pinch_click_threshold:
                    if not click_triggered:
                        pyautogui.click()
                        click_triggered = True
                else:
                    click_triggered = False

            # ======================
            # (Scroll)
            # ======================
            elif scroll_mode:
                if thumb_extended:
                    pyautogui.scroll(20)
                else:
                    pyautogui.scroll(-20)

            # ======================
            # (Reset Mode)
            # ======================
            elif reset_mode:
                mouse_x, mouse_y = 100, 100
                pyautogui.moveTo(mouse_x, mouse_y, duration=0.1)
                reset_mode = False

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (int(ix), int(iy)), 8, (255, 0, 0), cv2.FILLED)
    else:
        prev_x, prev_y, prev_z = None, None, None

    cv2.imshow("AI Smart Mouse - 3D Gestures", frame)# you can change the title of the window to whatever you want
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#you can fork this code and edit it as you want, but please keep the credits and the license.


