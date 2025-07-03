import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silent log TensorFlow

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from deepface import DeepFace

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)

cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    fingers = []

    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

gesture_state = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_face = face_detection.process(rgb)
    results_hands = hands.process(rgb)

    ih, iw, _ = frame.shape
    panel_width = 300
    panel = np.zeros((ih, panel_width, 3), dtype=np.uint8)
    panel[:] = (10, 10, 10)

    face_count = 0
    hand_count = 0
    finger_count = 0
    ekspresi = "-"
    umur_est = "-"

    # Face detection
    if results_face.detections:
        face_count = len(results_face.detections)
        for detection in results_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(iw, x + w), min(ih, y + h)
            face_crop = frame[y1:y2, x1:x2]

            try:
                face_crop_resized = cv2.resize(face_crop, (224, 224))
                analysis = DeepFace.analyze(
                    face_crop_resized,
                    actions=["age", "emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                    prog_bar=False
                )
                umur_est = str(analysis.get("age", "-"))
                ekspresi = analysis.get("dominant_emotion", "-")
            except:
                umur_est = "-"
                ekspresi = "-"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Exp: {ekspresi}", (x, y + h + 25), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {umur_est}", (x, y + h + 50), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2)

    # Hands
    if results_hands.multi_hand_landmarks:
        hand_count = len(results_hands.multi_hand_landmarks)
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

            finger_count = count_fingers(hand_landmarks)

            # Gesture control YouTube
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            thumb_x, thumb_y = int(thumb_tip.x * iw), int(thumb_tip.y * ih)
            index_x, index_y = int(index_tip.x * iw), int(index_tip.y * ih)
            distance = np.linalg.norm(np.array([thumb_x, thumb_y]) - np.array([index_x, index_y]))

            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)
            cv2.circle(frame, (thumb_x, thumb_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (index_x, index_y), 5, (0, 0, 255), -1)

            if distance < 40:
                cv2.putText(frame, "Gesture: PLAY", (10, 160), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                if gesture_state != "play":
                    pyautogui.press('space')
                    gesture_state = "play"
            else:
                cv2.putText(frame, "Gesture: PAUSE", (10, 160), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                if gesture_state != "pause":
                    pyautogui.press('space')
                    gesture_state = "pause"

    # Panel
    cv2.putText(panel, "[ STATUS ]", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    cv2.putText(panel, f"Faces: {face_count}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
    cv2.putText(panel, f"Hands: {hand_count}", (10, 110), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
    cv2.putText(panel, f"Fingers: {finger_count}", (10, 150), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
    cv2.putText(panel, f"Expression: {ekspresi}", (10, 190), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
    cv2.putText(panel, f"Age: {umur_est}", (10, 230), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
    cv2.putText(panel, "[ESC] EXIT", (10, ih - 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1)

    combined = np.hstack((frame, panel))
    cv2.imshow("Cyber Fast Detector", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
