#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:38:38 2017

@author: guilherme
"""

import csv 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D






lines = []
with open("/home/guilherme/data/driving_log.csv") as csvfile:
    reader=csv.reader(csvfile)
    
    for line in reader:
        lines.append(line)
        
images=[]
measurements=[]
for line in lines:
    images.append(cv2.imread(line[0]))#Center Image
    images.append(cv2.imread(line[1]))#Left Image
    images.append(cv2.imread(line[2]))#Right Image
    measurements.append(float(line[3]))#Center Image Angle
    measurements.append(float(line[3])+0.12)#Left Image offset angle
    measurements.append(float(line[3])-0.12)#Right Image offset angle
    
####Augmentation Data####
aug_images,aug_measurements = [],[]
for image,measurement in zip(images,measurements):
    aug_images.append(image)
    aug_images.append(cv2.flip(image,1))####To a flipped image
    aug_measurements.append(measurement)
    aug_measurements.append(measurement*-1)###Is setted an inverted angle value 
    
X_train=np.array(aug_images)
y_train=np.array(aug_measurements)

######Model Start Here#################################################

model=Sequential()
model.add(Lambda(lambda x:x/255 -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))#Cropping the TOP and the BOTTOM of the image.

#####Nvidea AutoPilot ConvNet ###############################################
model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))


model.compile(loss="mse",optimizer="adam") #Adam Opmizaer, no Learning rate set mannualy

#Training set splitted in a rate of 20% for validation
historic=model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=3,verbose=1)

print(historic.history.keys())
plt.plot(historic.history['loss'])
plt.plot(historic.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save("model.h5")





    
    
    
        
        
