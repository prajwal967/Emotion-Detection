import pandas as pd
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
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
K.set_image_dim_ordering('tf')
import joblib
K.image_data_format()

img_rows, img_cols = 48, 48

data = pd.read_csv(filepath)
data = data.drop(data.index[len(data)-1])

#Convert the data format from the original format to a 48x48 array.
#Original format is the pixel values stored in a string, where each value is seperated by a space.
def convert_data(data):
    
    X_train = data['pixels'][0].split(' ')
    X_train = np.asarray(X_train).astype('float32')
    X_train = X_train.reshape(1,48,48)
    
    Y_train = data['emotion']
    Y_train = np.asarray(Y_train)
    
    for i in range(1, data.shape[0]):
        
        x = data['pixels'][i].split(' ')
        x = np.asarray(x).astype('float32')
        x = x.reshape(1,48,48)
        X_train = np.concatenate((X_train,x), axis=0)

    
    return [X_train, Y_train]

def get_mean(X):
    
    mean = np.mean(X)
    
    return mean

def get_std(X):
    
    std = np.std(X)
    return std

#X, Y = convert_data(data)

#Save the converted dataset
#joblib.dump(X, 'X.pkl')
#joblib.dump(Y, 'Y.pkl')

#Load the saved dataset
X = joblib.load('X.pkl')
Y = joblib.load('Y.pkl')

def convert_img(img):
    img = imgToarr(img)

#Function converts the image to a numpy array
def imgToarr(img):
    
    return numpy.array(img)

#Resizes the image to the specified dimensions
def resize(img,x,y):
    
    return img.resize((x,y),Image.ANTIALIAS)

#Converting a N-Dimensional array to 1-D array
def reshape_1D(arr):
    
    return arr.ravel()

def list_to_numpyarray(convert_list):
    
    return np.asarray(convert_list)


X_train = X[0:28709]
Y_train = Y[0:28709]
X_test = X[28709:]
Y_test = Y[28709:]

Y_train = Y_train.reshape(Y_train.shape[0],1)
Y_test = Y_test.reshape(Y_test.shape[0],1)


X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

nb_classes = 7
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

#Initializing the values for the convolution neural network
nb_epoch = 100
batch_size = 50
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 4
# convolution kernel size
nb_conv = 3

model = Sequential()

model.add(Convolution2D(64, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

filepath="emotion1.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train, Y_train, batch_size=batch_size,           nb_epoch=nb_epoch, callbacks=callbacks_list,          validation_data=(X_test, Y_test),shuffle=True)


