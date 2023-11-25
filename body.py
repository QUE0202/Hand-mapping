import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:

    start_time = None
    closed_hand_duration = 0
    closed_hand_threshold = 10  # 10 sekundy zamkniętej dłoni

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Śledzenie pozy postaci
        pose_results = pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Śledzenie dłoni
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Sprawdzanie, czy dłoń jest zamknięta
                is_hand_closed = landmarks.landmark[mp_hands.HandLandmark.WRIST].y > \
                                 landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

                if is_hand_closed:
                    if start_time is None:
                        start_time = time.time()
                    else:
                        closed_hand_duration = time.time() - start_time
                        if closed_hand_duration >= closed_hand_threshold:
                            print("Zamknięta dłoń utrzymuje się przez 10 sekund. Zamykanie programu.")
                            cap.release()
                            cv2.destroyAllWindows()
                            exit()  # opcjonalnie, można użyć break, aby zakończyć pętlę

                else:
                    start_time = None  # Resetuj czas, gdy dłoń jest otwarta

            # Czytanie gestów
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Analiza gestów
                fingers_up = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y <
                              hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
                              hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y <
                              hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,
                              hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y <
                              hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y,
                              hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y <
                              hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y]

                # Reakcja na zamkniętą dłoń
                if all(fingers_up):
                    cv2.putText(frame, "Wszystkie palce uniesione", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif not any(fingers_up):
                    cv2.putText(frame, "Wszystkie palce opuszczone", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Pose and Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
