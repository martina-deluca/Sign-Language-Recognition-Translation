import numpy as np
import pandas as pd
import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def apply_model(frames):
    # Define parameters
    model = keras.models.load_model('model_parameters')
    
    # Get label mapping
    label_mapping_model = pd.read_csv('labels_for_the_model.csv', header=None)
    label_mapping_model = label_mapping_model.rename(columns={0: "label", 1: "word"})
    label_mapping_model['word'] = label_mapping_model['word'].astype(str)
    label_mapping_model = label_mapping_model.set_index('label')
    num_features = 126
    num_target_frames = 58

    # Filter landmark list and keep only 58 frames
    frames = [i for i in frames if i is not None]
    filtered_frames = frames[0:58]

    # Re-arrange frame structure
    resulting_frames = []
    for frame in filtered_frames:
        datapoints = []
        for datapoint in frame[0].landmark:
            datapoints.append(datapoint.x)
            datapoints.append(datapoint.y)
            datapoints.append(datapoint.z)
        resulting_frames.append(datapoints)

    # Create mask and features from frames' landmarks
    frame_mask = np.zeros(shape=(1, num_target_frames,), dtype="bool")
    frame_features = np.zeros(shape=(1, num_target_frames, num_features), dtype="float32")
    for i, frame in enumerate(resulting_frames):
        frame_mask[0, i] = 1  # 1 = not masked, 0 = masked
        frame_features[0, i, 0:len(frame)] = frame

    # Predict label
    probabilities = model.predict([frame_features, frame_mask])
    predicted_label = label_mapping_model.loc[np.argmax(probabilities)]

    return predicted_label['word']
