#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import keras
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from generator import DataGenerator
from models import testModel
from models import mclass2mlabel
from keras.utils import plot_model

def save_shuffled_df(train, val, test):
    """joined = np.vstack((np.vstack((X_tr, y_tr)).T, np.vstack((X_val, y_val)).T, np.vstack((X_ts, y_ts)).T))
    shuffled_df = pd.DataFrame(data={'Fpath': joined[:,0], 'Labels': joined[:,1]})
    shuffled_df.to_csv('shuffled.csv',index=False)"""
    return None

def main(args):
    ## Data prep
    seed = args.seed
    df = pd.read_csv('train_data_190726.csv', usecols=['Fpath', 'Labels'])
    df = df[~((df.Labels=='unkw') | (df.Labels=='leg'))] # shape: (8785, 2)
    df['Labels'] = df['Labels'].apply(lambda x:x.split(';'))
    X, y = df.Fpath.values, df.Labels.values
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, stratify=y, train_size=0.8, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, stratify=y_tr, train_size=0.8, random_state=seed)

    gen_params = {'dim': (512,512), 'batch_size': 16, 'shuffle': True,
                  'n_channels':1, 'multi': True}
    gen_tr = DataGenerator(X_tr, y_tr, **gen_params)
    gen_val = DataGenerator(X_val, y_val, **gen_params)
    #save_shuffled_df()

    ## Build model
    model = mclass2mlabel(keras.applications.inception_v3.InceptionV3(weights=None, input_shape=(512,512,1)), n_class=gen_tr.n_class)
    #model = testModel(n_class=gen_tr.n_class, multi=True).model
    optimizer = optimizers.rmsprop(lr=1e-5, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file=model.name+'.png')

    if args.train:
        weight_f = 'weights/190921_190726_multi_TestModel_{epoch:02d}.h5' # weights/{train_date}_{data_date}_multi_{model.name}_{epoch:02d}.h5'
        chkpoint = keras.callbacks.ModelCheckpoint(weight_f, save_weights_only=True, period=20)
        logger = keras.callbacks.CSVLogger('loss_file.csv')
        model.fit_generator(gen_tr, validation_data=gen_val, epochs=100, verbose=1, use_multiprocessing=True, workers=4, callbacks=[chkpoint, logger])

    if args.eval:
        model.load_weights(args.weight)
        gen_ts = DataGenerator(X_ts, batch_size=16, shuffle=False)
        preds = model.predict_generator(gen_ts, use_multiprocessing=True, verbose=1)
        preds_bool = (preds > 0.5)
        eval = multilabel_confusion_matrix(gen_tr.mlb.transform(y_ts), preds_bool.astype(int), samplewise=True)
        np.where(eval[:,0,0]+eval[:,1,1] != gen_tr.n_class)[0]
        X_ts[(eval[:,0,1] != 0) | (eval[:,1,0] != 0)]

        prediction = []
        for sample in preds_bool:
            l = [gen_tr.classes[i] for i, v in enumerate(sample) if v]
            if len(l) > 0: prediction.append(','.join(l))
            else: prediction.append('unkw')

            results = pd.DataFrame({'Fpath':X_ts, 'Truth':y_ts, 'Predict':prediction})
            results.to_csv('results.csv',index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', default=42)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--data', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--weight', default='weights/190805_190726_multi_TestModel_20.h5')
    main(parser.parse_args())
