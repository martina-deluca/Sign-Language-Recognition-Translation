# Sign-Language-Recognition-Translation
## Author
|Name|LinkedIn|
|-----|-----|
|Chih-Hua Chang|www.linkedin.com/in/chihhuachang1028|
|Martina De Luca|www.linkedin.com/in/martina-de-luca-barcello/|

## Objective

The goal of this project is to build a computer vision application capable of capturing video of a person signing a word in one sign language and output the translation of that word to the second sign language. In addition, the program will also display the written form of the word in both languages to validate the results more easily.

For example, using a video of a person signing in Argentinian Sign Language, the program generates the Spanish text, translates this word to English, and displays a video of the corresponding sign in American Sign Language.

## This repository includes:

**Code implementation:**
  - **Sign_Detection.py:** This file implements the Sign Detection module, it takes the input videos in LSA and extracts the features. 
  - **RNN_model.ipynb:** This file implements the Sign Recognition module, it uses the features extracted in the previous module to train the RNN. 
  - **Results_Preparation.py:** This file implements the Results Preparation module and uses the previosuly trained model to predict the word signed in a new input video.
  - **apply_model.py:** This is an auxiliary function for the Results Preparation module. 
  
**Data sets:**
  - **Input:** LSA64: A Dataset for Argentinian Sign Language (http://facundoq.github.io/datasets/lsa64/).
  - **Output:** WLASL (World Level American Sign Language) (https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed).

**Additional files:**
  - **LSA_label_transformation.csv and label_mapping.csv:** These files include mappings used in the code.

**Intermediate files:**
  - **landmarks.npy:** File output from the Sign Detection module, contains the features extracted for each video to be used in the Sign Recognition module.
  - **labels.npy:** File output from the Sign Detection module, contains the labels mapped for each video to be used in the Sign Recognition module.
  - **model parameters folder:** This folder contains all the details for the trained model from the Sign Recognition module, to be used in the final module.
  - **labels_for_the_model.csv:** This file contains a mapping between the number of the category predicted by the neural network model and the word it is predicted.
