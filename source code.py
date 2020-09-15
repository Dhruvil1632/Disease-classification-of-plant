

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from google.colab import drive
drive.mount('/content/drive')

location = '/content/drive/My Drive/new_images/plantvillage/PlantVillage'

labels = []
images_data = []
size = 700
for folder in os.listdir(location):
    #read subfolder path
    subfolder_path= os.path.join(location,folder)
    for images in os.listdir(subfolder_path):
        #read images path
        image_path = os.path.join(subfolder_path , images)
        img_array = cv2.imread(image_path)
        #resizing images
        resize_im = cv2.resize(img_array,(size,size))
        #normalize images
        resize_im = resize_im/255
        images_data.append(resize_im)
        labels.append(folder)

print("total number  of images : " ,len(images_data))
print("total number of labels : " , len(labels))

conv_labels = CountVectorizer()
onehot_labels = conv_labels.fit_transform(labels)
onehot_labels = onehot_labels.toarray()

conv_labels.inverse_transform(onehot_labels[1])

print("output shape must be :" , onehot_labels[1].shape[0])

x_train , x_test , y_train , y_test =  train_test_split(images_data , onehot_labels , test_size=0.15 , random_state = 2)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train  =  np.array(y_train)
y_test = np.array(y_test)

print("total training images :" , len(x_train))
print("total testing images :" , len(x_test))
print("shape of train dataset : " , x_train.shape)
print("shape of one image :" , x_train[1].shape)
print("shape of label dataset :" , y_train.shape)
print("shape of one label : " , y_train[1].shape)

classifier = Sequential()
# creating convolution layer
classifier.add(Conv2D(64,3,3,  input_shape = x_train[1].shape , activation ="relu" ))
# pooling 
classifier.add(MaxPooling2D(pool_size = (2,2)))
# creating convolution layer
classifier.add(Conv2D(32,3,3, activation ="relu" ))
# pooling 
classifier.add(MaxPooling2D(pool_size = (2,2)))
#flattening
classifier.add(Flatten())
#full connection
classifier.add(Dense(units = 15 , activation ="relu"))                 #units = shape of first onehot_label
classifier.add(Dense(units = 15 , activation = "sigmoid"))
classifier.compile(optimizer = "Adam" , loss = "binary_crossentropy" , metrics = ['accuracy'])
classifier.fit(x_train , y_train , batch_size = 20 , epochs = 5)
#y_test = conv_labels.inverse_transform(y_test)

classifier.summary()

def arraytolist(y_test):
    temp = []
    for i in y_test:
        sample1 = str(i)
        temp.append(sample1.split("'")[1])
    return temp
def accuracy_score(new_y_test , prediction):
    count = 0
    for i in range(0,len(new_y_test)):
        if new_y_test[i] == prediction[i]:
            count = count + 1
    acc_score = (count/len(prediction))*100
    return acc_score

#inverse reverse of onehot encoding
new_y_test = arraytolist(y_test)
#prediction of model
prediction = classifier.predict(x_test)

label_prediction = []
#process predicted onehot encoded data into meaning ful inspect
for i in range(0,len(prediction)):
  temp = []
  for i in prediction[i]:
    temp.append(float(i))
  temp = np.array(temp)   
  print(temp)
  # inverse transform of prediction to get original output
  prediction_ = conv_labels.inverse_transform(temp)
  label_prediction.append(prediction_)
label_prediction = arraytolist(label_prediction)
#determining accuray score
accuracy = accuracy_score(new_y_test , label_prediction)
print("accuracy score :" ,accuracy)

