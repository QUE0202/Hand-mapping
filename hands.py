import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

closed_frames = 0
threshold_seconds = 10  # liczba sekund, po których dłoń musi być zamknięta

start_time = None

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                thumb_to_index_distance = cv2.norm(
                    (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])),
                    (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
                )

                if thumb_to_index_distance < 50:
                    cv2.putText(frame, 'Closed', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    closed_frames += 1

                    if closed_frames == 1:
                        start_time = time.time()
                    elif closed_frames >= int(
                            threshold_seconds / 0.1) and time.time() - start_time >= threshold_seconds:
                        cv2.destroyAllWindows()
                        cap.release()
                        exit(0)
                else:
                    cv2.putText(frame, 'Open', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    closed_frames = 0
                    start_time = None

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
