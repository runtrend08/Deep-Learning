# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 23:29:48 2018

@author: 13913
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras import backend as K

from keras.models import Model
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
'''
'''
config = tf.ConfigProto()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)  
config.gpu_options.allow_growth = True
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
'''


batch_size = 128
num_classes = 15
epochs = 12

'''
第一步：导入数据集-ISCX
'''






## Importing the dataset
#dataset = pd.read_csv('D:\\pcapcsv\\all.csv')
#dataset = pd.read_csv('C:\\csv2\\merge-data\\all_matlab.csv') # 15万条不平衡数据集

#dataset = pd.read_csv('C:\\csv2\\merge-data\\balanced\\All_balance_15.csv')  #平衡数据集 7万条
dataset = pd.read_csv('.\\All_unbalance_15.csv')  #不平衡数据集 20万条


print (dataset.shape)

# Importing the dataset
#labelset = pd.read_csv('C:\\csv2\\label\\Label_15_app.csv') #15万条不平衡数据集

#labelset = pd.read_csv('C:\\csv2\\label\\balanced\\Label_15_balanced.csv')      #平衡数据集label
labelset = pd.read_csv('.\\Label_15_imbalanced.csv')      #不平衡数据集label

print (labelset.shape)


#X = dataset.iloc[:, 0:1479].values
#y = labelset.iloc[:, 0].values
X = dataset
y = labelset



#y = np_utils.to_categorical(y, 3)

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

y_train_transpose = np.transpose(y_train)
y_train_transpose = y_train_transpose.values
y_test_transpose = np.transpose(y_test)
y_test_transpose = y_test_transpose.values

Y_train = np_utils.to_categorical(y_train_transpose[0])
Y_test = np_utils.to_categorical(y_test_transpose[0])

print('******** Y_TRAIN ')
print(Y_train.shape)
print(Y_train)

print('******** Y1_Test ')
print(Y_test.shape)
print(Y_test)


Y1_train = Y_train[:,1:16]
print('******** Y1_TRAIN ')
print(Y1_train.shape)
print(Y1_train)
Y1_test = Y_test[:,1:16]
print('******** Y1_Test ')
print(Y1_test.shape)
print(Y1_test)



#Convert data type and normalize valuesPython
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255




inp =  Input(shape=(1480, 1))

x = Conv1D(128, 5, activation='relu')(inp)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(num_classes, activation='softmax')(x)

model = Model(inp, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

'''
增加TensorBoard相关记录
'''
from keras.callbacks import ModelCheckpoint, TensorBoard

checkpointer = ModelCheckpoint(filepath=".\model\\CNN_imbalanced\\CNN_model_15APP_gpu_ib.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='.\\tblog\\CNN\\CNN_imbalanced',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)


# happy learning!
X1_train = np.expand_dims(X_train, axis=2)
X1_test = np.expand_dims(X_test, axis=2)


model.fit(X1_train, Y1_train, epochs=100, batch_size=256)
# fit_history = model.fit(X1_train, Y1_train, validation_data=(X1_test, Y1_test),
#           epochs=100, batch_size=256, callbacks=[checkpointer, tensorboard]).history



print(model.summary())




scores = model.evaluate(X1_test, Y1_test, verbose=1) 
print("CNN Accuracy: ", scores[1])


'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

'''


import seaborn as sns
from pylab import rcParams

pred = model.predict(X1_test)

y_classes = pred.argmax(axis = -1)

y_true = Y1_test.argmax(axis = -1)

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_true,y_classes)
print(conf_matrix)

sns.set(style='whitegrid', palette='muted', font_scale=2.4)
sns.set_style("white")
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["aim_chat", "email","facebook", "gmail","hangout", "ICQ","netflix", "scpDown","sftpDown", "skype","spotify", "torTwitter","vimeo", "voipbuster","youtube"]

plt.figure(figsize=(24, 24))
sns.heatmap(conf_matrix, cmap="Oranges", xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", linewidths=0.2, linecolor = 'black', cbar=False);
plt.title("Traffic Classification Confusion matrix (CNN method)")
plt.ylabel('application traffic samples')
plt.xlabel('application traffic samples')
plt.show()

