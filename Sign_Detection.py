# Import libraries
import cv2
import mediapipe as mp
from os import listdir
import pathlib
import pandas as pd
from tqdm import tqdm
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Get list of videos from folder
path = 'Filtered Data/LSA/'
list_files = [f for f in listdir(path) if pathlib.Path(f).suffix == '.mp4']

# Create list with mapping for labels based on video code (first three letters)
label_df = pd.read_csv('LSA_label_transformation.csv', header=None)
label_df = label_df.rename(columns={0: "word", 1: "label"})
label_df['label'] = label_df['label'].astype(str)
label_df['label'] = label_df['label'].apply(lambda x: x.zfill(3))

# For each input video get number of frames
list_num_frames = []
for video in list_files:
    cap = cv2.VideoCapture(path+video)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    list_num_frames.append(num_frames)
list_num_frames.sort()
set(list_num_frames)
num_target_frames = min(list_num_frames)

# Create lists for saving results: data and labels
list_landmarks = []
list_labels = []

# Get landmarks for each video in the list
for file in tqdm(list_files):
    cap = cv2.VideoCapture(path+file)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    step_size = int(num_frames//num_target_frames)
    landmarks_frames = []
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        current_frame = 0
        while current_frame < int(num_target_frames*step_size):
            ret, frame = cap.read()
            if current_frame % step_size == 0:
                # BGR 2 RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Flip on horizontal
                image = cv2.flip(image, 1)

                # Set flag
                image.flags.writeable = False

                # Detections
                results = hands.process(image)

                # Set flag to true
                image.flags.writeable = True

                # RGB 2 BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Rendering results
                if results.multi_hand_landmarks:
                    for num, hand in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                                 )

                landmarks_frames.append(results.multi_hand_landmarks)
            current_frame += 1

    cap.release()
    cv2.destroyAllWindows()
    if len(landmarks_frames) == 0:
        print('Error - No landmarks were dectected for this video:',file)
    # Add results to lists for this video
    list_landmarks.append(landmarks_frames)
    list_labels.append(label_df.loc[label_df['label']==file[0:3],'word'].iloc[0])

# Print results to verify numbers
print('Length of file list:', len(list_files))
print('Length of landmark list:', len(list_landmarks))
print('Length of label list:', len(list_labels))
print('Set of label values:', set(list_labels))
print('Length of set of label values:', len(set(list_labels)))

# Save to files for next module
with open('landmarks.npy', 'wb') as f:
    np.save(f, list_landmarks)
with open('labels.npy', 'wb') as f:
    np.save(f, list_labels)