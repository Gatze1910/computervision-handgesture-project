import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

gesture = 'Telefon' #change here the gesture: Faust, Thumb-Up, Thumb-Down, Open-Hand, Peace, Telefon
samples_per_gesture = 100 #change here the amount of samples you like to take

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

os.makedirs(f'data/{gesture}', exist_ok=True)
os.makedirs('sample', exist_ok=True) 
sample_count = 0
    
while sample_count < samples_per_gesture:
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
                
            np.save(f'data/{gesture}/b_l_{sample_count}_n_nl.npy', landmarks)

            # save each 10. image for paper dokumentation
            if sample_count % 10 == 0: 
                cv2.imwrite(f'sample/{gesture}_b_l_{sample_count}_n_nl.png', frame)

            sample_count += 1
            print(f'Gesture: {gesture}, Sample: {sample_count}')
    
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    

cap.release()
cv2.destroyAllWindows()
