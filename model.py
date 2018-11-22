#import Libraries
import os
import csv
import cv2
import sklearn
import random
import numpy as np
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from itertools import chain


#Read Driving Data from Track 1
samples_t1 = []
with open("../CarND-Behavioral-Cloning-P3/data_t1/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples_t1.append(line)

#Split Track 1 data into training and validation set
train_samples_t1, validation_samples_t1 = train_test_split(samples_t1, test_size=0.2)

#Read Driving Data from Track 2
samples_t2 = []
with open("../CarND-Behavioral-Cloning-P3/data_t2/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples_t2.append(line)

#Split Track 2 data into training and validation set
train_samples_t2, validation_samples_t2 = train_test_split(samples_t2, test_size=0.2)

#Generator definition to avoid storing augmented data in memory
def generator(samples, batch_size=32, track=1):
    num_samples = len(samples)
    while 1:    #Infinite loop, generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            #Arrays to store images and steering angles
            images = []
            angles = []

            #Process multiple images at once
            for batch_sample in batch_samples:
                #sets which track is being used and where the images are located
                if track is 1:
                    center_path = '../CarND-Behavioral-Cloning-P3/data_t1/IMG/' + batch_sample[0].split('/')[-1]
                else:
                    center_path = '../CarND-Behavioral-Cloning-P3/data_t2/IMG/' + batch_sample[0].split('/')[-1]

                #get the steering angle for the image and calcuate the inverse steering angle
                angle = float(batch_sample[3])
                flipped_angle = angle * -1.0

                #get the image and convert to RGB, then create a horizontally flipped image
                image = cv2.imread(center_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                flipped_image = cv2.flip(image, 1)

                #Add the image and steering angle to the arrays
                images.append(image)
                angles.append(angle)

                #Readd the image and introduce variation in the steering data
                images.append(image)
                angles.append(angle + 0.2)

                images.append(image)
                angles.append(angle - 0.2)

                #Add the inverse of the image to help the model generalize
                images.append(flipped_image)
                angles.append(flipped_angle)

            #Convert array to numpy arrays
            X_train = np.array(images)
            y_train = np.array(angles)
            #Generator's equivalent of return
            yield sklearn.utils.shuffle(X_train, y_train)

##Create training set generator
#Track 1
train_generator = generator(train_samples_t1, batch_size=128, track=1)
#Track 2
#train_generator = generator(train_samples_t2, batch_size=16, track=2)
#Combine tracks 1 & 2
#train_generator = chain(train_generator_t1,  train_generator_t2)

##Create validation set generator
#Track 1
validation_generator = generator(validation_samples_t1, batch_size=128, track=1)
#Track 2
#validation_generator = generator(validation_samples_t2, batch_size=16, track=2)
#Combine tracks 1 & 2
#validation_generator = chain(validation_generator_t1, validation_generator_t2)

#import libraries for Keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#Start the model in sequential mode
model = Sequential()

#Crop the top of the image to remove images of just sky
model.add(Cropping2D(cropping=((65,0), (0,0)), input_shape=(160,320,3)))
#Normalized pixel values from 0 to 2 and mean center values from -1 to 1
model.add(Lambda(lambda x: (x/127.5) - 1.0))


#NVIDIA Architecture
model.add(Convolution2D(24,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(64,(3,3),strides=(1,1),activation='relu'))
model.add(Convolution2D(64,(3,3),strides=(1,1),activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#Compile the model, using an Adam optimizer to dynamically tune hyperparameters
model.compile(loss='mse', optimizer='adam')

#Run the model
#Track 1
model.fit_generator(train_generator, steps_per_epoch=len(train_samples_t1), validation_data=validation_generator, validation_steps=len(validation_samples_t1), epochs=1, verbose=1)
#Track 2
#model.fit_generator(train_generator, steps_per_epoch=len(train_samples_t2), validation_data=validation_generator, validation_steps=len(validation_samples_t2), epochs=2, verbose=1)
#Track 1 & Track 2
#model.fit_generator(train_generator, steps_per_epoch=len(train_samples_t1 + train_samples_t2), validation_data=validation_generator, validation_steps=len(validation_samples_t1 + validation_samples_t2), epochs=2, verbose=1)

#Save the model
model.save('model.h5')
exit()
