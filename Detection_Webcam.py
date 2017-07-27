
import cv2
import pandas as pd
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
import skimage
import glob
import numpy
from sklearn.metrics.pairwise import cosine_similarity
from skimage.measure import compare_ssim
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import mutual_info_score
from PIL import Image
from keras.models import load_model
from keras import backend as K
import heapq
import matplotlib.pyplot as plt

faceCascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

video_capture = cv2.VideoCapture(0)

m = load_model('emotion1.h5')

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for face in faces:
        x,y,w,h = face
        roi_gray = gray[y:y+h, x:x+w]
        resize_img = cv2.resize(roi_gray,(48,48))
        
        img_arr = np.array(resize_img)
        img_arr = img_arr.astype('float32')
        prediction = m.predict(img_arr)



        labels = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
        predictions_arr = m.predict(img_arr)
        predictions = predictions_arr[0].tolist()
        max_index = heapq.nlargest(3, range(len(predictions)), key=predictions.__getitem__)
        print('Emotion Rank')
        print('1.',labels[max_index[0]])
        print('2.',labels[max_index[1]])
        print('3.',labels[max_index[2]])

        if (labels[max_index[0]]=="Happy"):
            cv2.putText(frame,"Happy", (x,y), cv2.FONT_ITALIC, 3, (0,0,255))

        elif (labels[max_index[0]]=="Angry"):
            cv2.putText(frame,"Angry", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255))

        elif (labels[max_index[0]]=="Disgust"):
            cv2.putText(frame,"Disgust", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255))

        elif (labels[max_index[0]]=="Fear"):
            cv2.putText(frame,"Fear", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255))

        elif (labels[max_index[0]]=="Sad"):
            cv2.putText(frame,"Sad", (x,y), cv2.FONT_HERSHEY_SIMPLEX,3, (0,0,255))

        elif (labels[max_index[0]]=="Surprise"):
            cv2.putText(frame,"Surprise", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255))

        else:
            cv2.putText(frame,"Neutral", (x,y), cv2.FONT_ITALIC,3, (0,0,255))


        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('Video', frame)
    # cv2.resizeWindow('Video',640,600)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()




