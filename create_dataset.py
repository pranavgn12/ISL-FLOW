import os
import cv2
import mediapipe as mp
import csv
import numpy as np

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Path to your dataset
DATA_DIR = './data'  # Structure: ./data/A/img1.jpg, ./data/B/img1.jpg...

# Prepare CSV file
with open('isl_keypoints.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    # Header: label, then 21*3 coords for Left hand, 21*3 for Right hand
    header = ['label'] + [f'L_x{i}' for i in range(21)] + [f'L_y{i}' for i in range(21)] + \
             [f'R_x{i}' for i in range(21)] + [f'R_y{i}' for i in range(21)]
    writer.writerow(header)

    labels = os.listdir(DATA_DIR)

    for label in labels:
        print(f"Processing Letter: {label}")
        folder_path = os.path.join(DATA_DIR, label)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                # Initialize empty arrays for Left and Right hands (21 points * 2 coords typically, but let's use x,y)
                # Actually simpler: Flatten everything to x, y. Z is often noisy in 2D images.
                # Let's use 42 values per hand (x,y) -> 84 features total.

                left_hand_data = [0] * 42
                right_hand_data = [0] * 42

                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # MediaPipe says "Left" for the hand that appears on the left of the image (viewer's left)
                    # or strictly by hand type. We rely on label.
                    hand_type = handedness.classification[0].label

                    # Extract coords
                    temp_data = []
                    for lm in hand_landmarks.landmark:
                        temp_data.append(lm.x)
                        temp_data.append(lm.y)

                    if hand_type == 'Left':
                        left_hand_data = temp_data
                    else:
                        right_hand_data = temp_data

                # Write to CSV
                writer.writerow([label] + left_hand_data + right_hand_data)

print("Data processing complete!")
