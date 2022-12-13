# Imports
import cv2
import os
import pandas as pd
import moviepy.editor as mpy
import mediapipe as mp
import time
from apply_model import apply_model
import argparse

# Live video or from library
parser = argparse.ArgumentParser()
'''
live_video = False  # change to True to use a live video
if live_video:
    path_input = 0
else:
    path_input = 'original_input_video_red.mp4'''

# Update values based on parameters input by the user
parser.add_argument("--live_video", help="True for live_video, False for pre-recorded video.", action="store_true", default=False)
parser.add_argument("--video", help="File name if pre-recorded video selected.", default="original_input_video_red.mp4")
args = parser.parse_args()

live_video = args.live_video
path_input = args.video

if live_video:
    path_input = 0

# Parameters
landmarks_frames = []
length = 3
num_target_frames = 58
original_input = 'original_input_video.mp4'
name_input_video = 'input_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Have mapping between english word and spanish word
label_mapping = pd.read_csv('label_mapping.csv', header=None)
label_mapping = label_mapping.rename(columns={0: "english", 1: "spanish"})
label_mapping['english'] = label_mapping['english'].astype(str)
label_mapping['spanish'] = label_mapping['spanish'].astype(str)
label_mapping = label_mapping.set_index('english')

# Capture live/library video, save video as input and landmarks for NN
cap = cv2.VideoCapture(path_input)
out = cv2.VideoWriter(name_input_video, fourcc, 30, (600, 400))
out_original = cv2.VideoWriter(original_input, fourcc, 30, (600, 400))
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        landmarks_frames.append(results.multi_hand_landmarks)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image2 = cv2.resize(image, (600, 400))
        out_original.write(image2)
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                          )
        cv2.imshow('Hand Tracking', image)
        image = cv2.resize(image, (600, 400))
        out.write(image)
        if cv2.waitKey(1) == ord('q'):
            break
# When everything done, release the capture
cap.release()
out.release()
out_original.release()
cv2.destroyAllWindows()

# Process in NN to identify the word (english label)
predicted_label_english = apply_model(landmarks_frames)

# find video in ASL
path_ASL = 'Filtered Data/ASL/videos/' + predicted_label_english + '/'
ASL_video = os.listdir(path_ASL)[1]

# Map english word to spanish word
predicted_label_spanish = label_mapping.loc[predicted_label_english]
predicted_label_spanish = predicted_label_spanish['spanish']

# Add the words to the video
def add_text_video(video_path, text, output_name):
    video = cv2.VideoCapture(video_path)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    out = cv2.VideoWriter(output_name, fourcc, frames//3, (600, 400))

    while (video.isOpened()):
        success, frame = video.read()
        if not success:
            break
        frame = cv2.resize(frame, (600, 400))
        cv2.putText(frame, text, (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        out.write(frame)

    video.release()
    out.release()

add_text_video(name_input_video, predicted_label_spanish, 'result_spanish.mp4')
add_text_video(path_ASL + ASL_video, predicted_label_english, 'result_english.mp4')

# Display words in both languages and ASL video
video1 = mpy.VideoFileClip('result_spanish.mp4').subclip(0, 0 + length)
video2 = mpy.VideoFileClip('result_english.mp4').subclip(0, 0 + length)

combined = mpy.clips_array([[video1, video2]])
combined.write_videofile('result.mp4', logger=None)

cap = cv2.VideoCapture('result.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        time.sleep(1/fps)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
