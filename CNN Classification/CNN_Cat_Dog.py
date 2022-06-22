#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:54:38 2020
Predicting CAT or DOG
using Convolutional neural network
@author: utkarshkushwaha
"""

import os
os.chdir('/Users/utkarshkushwaha/Desktop/Spyderworkspace')

import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#initializing the CNN
classifier = Sequential()

#Step-1 convolution
classifier.add(Convolution2D(32,(3,3), input_shape = (64,64,3), activation = 'relu'))

#step-2 maxpooling
classifier.add(MaxPooling2D(pool_size =(2,2)))

#step-3 flattening
classifier.add(Flatten())

#step-4 full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))#units = output_dim same
classifier.add(Dense(output_dim = 1, activation ='sigmoid'))

classifier.summary()

#compile the CNN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

#fitting CNN to th images(Training and Testing )
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip =True, fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('cnn_imgs_dataset/training_set', target_size =(64,64), batch_size = 32, class_mode = 'binary')

test_dataset = test_datagen.flow_from_directory('cnn_imgs_dataset/test_set', target_size = (64,64), batch_size = 32, class_mode = 'binary')

classifier.fit_generator(training_set, samples_per_epoch = 200, nb_epoch = 10, validation_data = test_dataset, nb_val_samples=100)  
# classifier.fit_generator(training_set, samples_per_epoch = 2000, nb_epoch = 5, validation_data = test_dataset, nb_val_samples = 1000)
classifier.summary()           

#loss: 0.5265 - accuracy: 0.7325




#this will tke more time
'''
classifier.fit_generator(training_set, samples_per_epoch =8000, nb_epoch = 5, validation_data = test_dataset, nb_val_samples= 2000 )                                                                                                                
'''                                                          


## loading image and preprocessing

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('cnn_imgs_dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image

#addig img 3D to 3D-array
test_image = image.img_to_array(test_image)
test_image


#adding axis and making 4D
test_image = np.expand_dims(test_image, axis = 0)
test_image


#Now predicting
x = preprocess_input(test_image)
x

result = classifier.predict(x)
result

print(training_set.class_indices)

if result[0][0] == 1:
    
    prediction = 'Dog'
    
else:
    
    prediction = 'Cat'

print(prediction)

'''
True its dog
'''



#again testing its cat r dog 
test_image2 = image.load_img('cnn_imgs_dataset/single_prediction/cat_or_dog_3.jpg', target_size = (64,64))
test_image2

test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis =0)

x = preprocess_input(test_image2)
x

result = classifier.predict(x)
result
print(training_set.class_indices)

if result[0][0] == 1:
    prediction = 'Dog'
else:
    prediction = 'Cat'
print(prediction)

#its 85.43 % propability