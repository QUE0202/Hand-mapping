import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

closed_frames = 0
threshold_seconds = 10  # liczba sekund, po których postać musi być w określonym stanie

start_time = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Sprawdzanie warunków postaci
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            nose_to_left_shoulder_distance = cv2.norm(
                (int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0])),
                (int(left_shoulder.x * frame.shape[1]), int(left_shoulder.y * frame.shape[0]))
            )

            nose_to_right_shoulder_distance = cv2.norm(
                (int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0])),
                (int(right_shoulder.x * frame.shape[1]), int(right_shoulder.y * frame.shape[0]))
            )

            if nose_to_left_shoulder_distance < 100 or nose_to_right_shoulder_distance < 100:
                cv2.putText(frame, 'Closed', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                closed_frames += 1

                if closed_frames == 1:
                    start_time = time.time()
                elif closed_frames >= int(threshold_seconds / 0.1) and time.time() - start_time >= threshold_seconds:
                    cv2.destroyAllWindows()
                    cap.release()
                    exit(0)
            else:
                cv2.putText(frame, 'Open', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                closed_frames = 0
                start_time = None

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Pose Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
