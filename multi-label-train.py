#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import pydicom as dcm
import pickle
import keras
from keras_preprocessing.image import ImageDataGenerator
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from generator import DataGenerator
from models import testModel
from keras.utils import plot_model


## Data prep
seed = 42
df = pd.read_csv('train_data_190726.csv', usecols=['Fpath', 'Labels'])
df = df[~((df.Labels=='unkw') | (df.Labels=='leg'))] # shape: (8785, 2)
df['Labels'] = df['Labels'].apply(lambda x:x.split(';'))
X, y = df.Fpath.values, df.Labels.values
X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify=y, train_size=0.7, random_state=seed)
X_val, X_ts, y_val, y_ts = train_test_split(X_val, y_val, stratify=y_val, train_size=0.6, random_state=seed)

gen_params = {'dim': (512,512), 'batch_size': 16, 'shuffle': True,
              'n_channels':1, 'multi': True}
gen_tr = DataGenerator(X_tr, y_tr, **gen_params)
gen_val = DataGenerator(X_val, y_val, **gen_params)

"""joined = np.vstack((np.vstack((X_tr, y_tr)).T, np.vstack((X_val, y_val)).T, np.vstack((X_ts, y_ts)).T))
shuffled_df = pd.DataFrame(data={'Fpath': joined[:,0], 'Labels': joined[:,1]})
shuffled_df.to_csv('shuffled.csv',index=False)"""

## Build model
in_img = keras.layers.Input(shape=(512, 512, 1), dtype='float')
net = testModel(n_class=gen_tr.n_class, mode='sigmoid')
optimizer = optimizers.rmsprop(lr=1e-5, decay=1e-6)
net.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#plot_model(net.model, to_file='model.png')
net.model.load_weights('weights/190803_190726_multi_TestModel_100.h5')

## Train model
"""weight_f = 'weights/190803_190726_multi_TestModel_{epoch:02d}.h5'
chkpoint = keras.callbacks.ModelCheckpoint(weight_f, save_weights_only=True, period=20)
logger = keras.callbacks.CSVLogger('loss_file.csv')
net.model.fit_generator(gen_tr, validation_data=gen_val, epochs=100, verbose=1, use_multiprocessing=True, callbacks=[chkpoint, logger])"""

"""from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit(y_tr)
mlb.classes_
mlb.transform(y_tr[100:103])
y_tr[100:103]"""

## Predict output
gen_ts = DataGenerator(X_ts, batch_size=16, shuffle=False)
preds = net.model.predict_generator(gen_ts, use_multiprocessing=True, verbose=1)
preds.shape, X_ts.shape
#preds = net.model.predict(getPixelData(X_ts))
#preds_bool = (preds > 0.5)

"""model.load_weights('weights.h5')
ts_gen.reset()
pred = model.predict_generator(ts_gen, steps=STEP_SIZE_TEST, verbose=1)
pred_bool = (pred > 0.5)

predictions=[]
labels = tr_gen.class_indices
labels = dict((v,k) for k,v in labels.items())
for row in pred_bool:
    l = []
    for index, cls in enumerate(row):
        if cls:
            l.append(labels[index])
    predictions.append(','.join(l))
filenames = ts_gen.filenames
results = pd.DataFrame({'Filename':filenames, 'Predictions':predictions})
results.to_csv('results.csv',index=False)"""
