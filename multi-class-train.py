#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import keras
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from generator import DataGenerator
from models import testModel
import pydicom as dcm


## Read training data
train_df = pd.read_csv('train_data_190703_checked.csv')
train_df = train_df[~((train_df.Label=='UNKW') | (train_df.Label=='LEG'))]
#train_df.Label.value_counts()
train_df.Label, classes = pd.factorize(train_df.Label)
n_class = len(classes)
X, y = train_df.Fpath.values, train_df.Label.values
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, stratify=y, train_size=0.8, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, stratify=y_tr, train_size=0.8, random_state=42)

gen_params = {'dim': (512,512), 'batch_size': 16, 'n_class': n_class,
              'shuffle': True, 'n_channels':1}
gen_train = DataGenerator(X_tr, y_tr, **gen_params)
gen_val = DataGenerator(X_val, y_val, **gen_params)

## Build model
in_img = keras.layers.Input(shape=(512, 512, 1), dtype='float')
#model = keras.applications.vgg16.VGG16(input_tensor=in_img, weights=None, classes=n_class)
#model = keras.applications.inception_v3.InceptionV3(input_tensor=in_img, weights=None, classes=n_class)
#model.compile(loss='categorical_crossentropy', optimizer='adam')
net = testModel(n_class=n_class)
model = net.model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-6))

## Train model
epochs = 100
loss_file = open('loss_file.txt', 'w')
for epoch in range(1, epochs+1):
    hist = model.fit_generator(gen_train, validation_data=gen_val, epochs=1, verbose=1, use_multiprocessing=True, workers=4)
    loss_file.write('%s,%s\n' %(hist.history['loss'], hist.history['val_loss']))
    if epoch%20 == 0: model.save_weights('weights/190715_190703_checked_TestModel_%s.h5'%epoch)

loss_file.close()

## Evaluate model
model.load_weights('weights/190715_190703_checked_TestModel_100.h5')
gen_ts = DataGenerator(X_ts, batch_size=16, shuffle=False)
preds = model.predict_generator(gen_ts, use_multiprocessing=True, verbose=1)
confusion_matrix(y_ts, preds.argmax(axis=1))

map = {0:'CHST', 1:'ABDM', 2:'PELV', 3:'SHDR', 4:'NECK', 5:'HEAD'} # Labeled prediction
y_pred = [map[i] for i in preds.argmax(axis=1)]
y_true = [map[int(i)] for i in y_ts]
confusion_matrix(y_true, y_pred, labels=['HEAD', 'NECK', 'SHDR', 'CHST', 'ABDM', 'PELV'])

np.where([y_ts.astype(int) != preds.argmax(axis=1)])[1]
X_ts[y_ts != preds.argmax(axis=1)] # print out misclassifications

"""# Example result from data using directory and tag info only
test_df = pd.read_csv('train_data_SEMAR_ON_edited.csv')
test_df = test_df[~((test_df.Label=='PELVIS') | (test_df.Label=='HEAD') |
                     (test_df.Label=='FEMUR') | (test_df.Label=='NECK'))]
model.load_weights('weights/190709_train_data_190703_edited100.h5')
gen_ts = DataGenerator(test_df.Fpath.values, batch_size=32, shuffle=False)
preds = model.predict_generator(gen_ts, use_multiprocessing=True, verbose=1)
map = {0:'ABDOMEN', 1:'NECK', 2:'HEAD', 3:'CHEST'}
res = [map[i] for i in preds.argmax(axis=1)]
confusion_matrix(test_df.Label, res, labels=['ABDOMEN', 'NECK', 'HEAD', 'CHEST'])"""
