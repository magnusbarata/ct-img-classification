#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import keras
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from generator import DataGenerator
from models import testModel
import pydicom as dcm
import pickle

"""def generator(flist, labels, batch_size, n_class):
    b_features = np.zeros(shape=(batch_size, 512, 512, 1))
    b_labels = np.zeros(shape=(batch_size, n_class))

    while True:
        for i in range(batch_size):
            index = np.random.choice(len(flist), batch_size)
            b_features[i] = features[index]
            b_labels[i] = labels[index]
        yield b_features, b_labels"""


## Read training data
train_df = pd.read_csv('train_data_190703_checked.csv')
train_df = train_df[~((train_df.Label=='UNKW') | (train_df.Label=='LEG'))]
#train_df.Label.value_counts()
train_df.Label, classes = pd.factorize(train_df.Label)
n_class = len(classes)
X, y = train_df.Fpath.values, train_df.Label.values
X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify=y, train_size=0.7)
X_val, X_ts, y_val, y_ts = train_test_split(X_val, y_val, stratify=y_val, train_size=0.6)
train_data = np.array([[x,y] for x, y in zip(X_tr, y_tr)])
val_data = np.array([[x,y] for x, y in zip(X_val, y_val)])
test_data = np.array([[x,y] for x, y in zip(X_ts, y_ts)])
"""with open('train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open('val_data.pkl', 'wb') as f:
    pickle.dump(val_data, f)
with open('test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)"""

## Build model
in_img = keras.layers.Input(shape=(512, 512, 1), dtype='float')
#model = keras.applications.vgg16.VGG16(input_tensor=in_img, weights=None, classes=n_class)
#model = keras.applications.inception_v3.InceptionV3(input_tensor=in_img, weights=None, classes=n_class)
net = testModel(classes=n_class)
model = net.model
model.summary()
#model.compile(loss='categorical_crossentropy', optimizer='adam')
optimizer = Adam(lr=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
#model.load_weights('weights/190712_190703_checked_InceptionV3_100.h5')
model.load_weights('weights/190715_190703_checked_TestModel_100.h5')

## Train model
gen_params = {'dim': (512,512), 'batch_size': 16, 'n_classes': n_class,
              'shuffle': True, 'n_channels':1}
gen_train = DataGenerator(X_tr, y_tr, **gen_params)
gen_val = DataGenerator(X_val, y_val, **gen_params)
epochs = 100
loss_file = open('loss_file.txt', 'w')
for epoch in range(1, epochs+1):
    hist = model.fit_generator(gen_train, validation_data=gen_val, epochs=1, verbose=1, use_multiprocessing=True)
    loss_file.write(str(hist.history['loss']) + ',' + str(hist.history['val_loss']) + '\n')
    #loss_list.append([hist.history['loss'], hist.history['val_loss']])
    if epoch%20 == 0: model.save_weights('weights/190715_190703_checked_TestModel_' + str(epoch) + '.h5')

loss_file.close()

## Evaluate model
def getPixelData(fnames, n_channels=1):
    pixels = []
    for f in fnames:
        ds = dcm.dcmread(f)
        pixel_array = ds.pixel_array
        pixels.append(pixel_array.reshape(pixel_array.shape+(n_channels,)))
    return np.array(pixels)

with open('test_data.pkl', 'rb') as f:
    test_file = pickle.load(f)
np.unique(test_file[:,1], return_counts=True)
test_data = getPixelData(test_file[:,0])
preds = model.predict(test_data)
map = {0:'CHST', 1:'ABDM', 2:'PELV', 3:'SHDR', 4:'NECK', 5:'HEAD'}
y_pred = [map[i] for i in preds.argmax(axis=1)]
y_true = [map[int(i)] for i in test_file[:,1]]
confusion_matrix(y_true, y_pred, labels=['HEAD', 'NECK', 'SHDR', 'CHST', 'ABDM', 'PELV'])
"""
test_df = pd.read_csv('train_data_SEMAR_ON_edited.csv')
test_df = test_df[~((test_df.Label=='PELVIS') | (test_df.Label=='HEAD') |
                     (test_df.Label=='FEMUR') | (test_df.Label=='NECK'))]
test_df.Label.value_counts()
test_data = getPixelData(test_df.Fpath.values)
test_data.shape
model.load_weights('weights/190709_train_data_190703_edited100.h5')
preds = model.predict(test_data)
map = {0:'ABDOMEN', 1:'NECK', 2:'HEAD', 3:'CHEST'}
res = [map[i] for i in preds.argmax(axis=1)]
confusion_matrix(test_df.Label, res, labels=['ABDOMEN', 'NECK', 'HEAD', 'CHEST'])"""
