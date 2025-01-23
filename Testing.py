import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

model = load_model(f'./B_Model_50epochs_acc_097.hdf5')
gestures = ['faust', 'peace', 'thumb-up', 'thumb-down', 'open-hand', 'telefon']

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).reshape(1, 63, 1)
            prediction = model.predict(landmarks)
            gesture_index = np.argmax(prediction)
            gesture = gestures[gesture_index]

            cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
