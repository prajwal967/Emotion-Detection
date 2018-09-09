# Emotion Detection

The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
 
## Project

The model is trained in the Detection-Kaggle.py and this model is used to detect the emotion in the Detection-Webcam.py. A convolution neural network is used to train the model (emotion1.h5), the backend that is used to train the model is a Tensorflow-GPU. Detection-Webcam.py uses OpenCV to activate the webcam and then uses the trained model to predict the user's emotion from the captured images on the webcam.

## Trained Model

The size of the trained model is too large to be uploaded on GitHub. 
To obtain the model run the Detection-Kaggle.py file. The model will be stored as emotion1.
This model will then be used t0 identify the emotion when the Detection_Webcam.py file is run

