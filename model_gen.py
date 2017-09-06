#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:38:38 2017

@author: guilherme
"""
import sklearn
import csv 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split

samples = []
with open("/home/guilherme/data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            aug_images,aug_measurements = [],[]
            
            for batch_sample in batch_samples:
                center = '/home/guilherme/data/IMG/'+batch_sample[0].split('/')[-1]
                left = '/home/guilherme/data/IMG/'+batch_sample[1].split('/')[-1]
                right = '/home/guilherme/data/IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(center)
                left_image = cv2.imread(left)
                right_image = cv2.imread(right)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(center_angle + 0.12) #left offset
                angles.append(center_angle - 0.12) #right offset

        		
            '''for image,measurement in zip(images,angles):
                aug_images.append(image)
                aug_images.append(cv2.flip(image,1))
                aug_measurements.append(measurement)
                aug_measurements.append(measurement*-1)'''

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=1)
validation_generator = generator(validation_samples, batch_size=1)

row, col, ch = 160, 320, 3  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255 - 0.5,
        input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

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


model.compile(loss='mse', optimizer='adam')
historic=model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=3,verbose=1)

print(historic.history.keys())
plt.plot(historic.history['loss'])
plt.plot(historic.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save("model.h5")









    
    
    
        
        
