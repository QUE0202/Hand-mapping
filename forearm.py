import cv2
import mediapipe as mp

import time
import platform

if platform.system() == "Darwin":
    cv2.VideoCapture(0).release()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Ustawienie mediapipe Hands na tryb statyczny
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

with hands, mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Śledzenie pozy postaci
        pose_results = pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))

            # Pobieranie współrzędnych łokcia i nadgarstka
            left_elbow = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_elbow = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Rysowanie linii przedramion
            cv2.line(frame, (int(left_elbow.x * frame.shape[1]), int(left_elbow.y * frame.shape[0])),
                     (int(left_wrist.x * frame.shape[1]), int(left_wrist.y * frame.shape[0])), (255, 0, 0), 2)
            cv2.line(frame, (int(right_elbow.x * frame.shape[1]), int(right_elbow.y * frame.shape[0])),
                     (int(right_wrist.x * frame.shape[1]), int(right_wrist.y * frame.shape[0])), (255, 0, 0), 2)

        # Śledzenie dłoni
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
