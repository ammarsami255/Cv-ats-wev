import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

img_width, img_height = 32, 32    
model_path = 'sign_language_model.h5'  

model = load_model(model_path)
labels = sorted(os.listdir('dataset/Gesture Image Data'))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
        
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            padding = 20
            x_min = max(x_min - padding, 0)
            y_min = max(y_min - padding, 0)
            x_max = min(x_max + padding, w)
            y_max = min(y_max + padding, h)
            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size != 0:
                img = cv2.resize(roi, (img_width, img_height))
                img = img.astype('float32') / 255.0  
                img = np.expand_dims(img, axis=0)  

                if img.shape[-1] == 1:  
                    img = np.repeat(img, 3, axis=-1)  

                predictions = model.predict(img)  
                class_idx = np.argmax(predictions)
                predicted_label = labels[class_idx]
                confidence = predictions[0][class_idx]
                
                if confidence > 0.3: 
                    cv2.putText(frame, f'{predicted_label} ({confidence*100:.2f}%)', 
                                (x_min, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                print(f'Predictions: {predictions}, Predicted Label: {predicted_label}, Confidence: {confidence}')

    cv2.imshow("Sign Language Translator", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
