#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
from utils import *

import pandas as pd
import numpy as np
import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from generator import DataGenerator
from models import *
# TODO: wrap normalize class (preprocess lib), n_channel integration

def normalize(img_array, a_min=None, a_max=None):
    if a_min is not None and a_max is not None:
        img_array = np.clip(img_array, a_min, a_max)
    return (img_array - np.amin(img_array)) / (np.amax(img_array) - np.amin(img_array))

def save_shuffled_df(train, test, val=None):
    if val is not None:
        joined = np.vstack((np.hstack((train[0], val[0], test[0])), np.hstack((train[1], val[1], test[1])))).T
    else:
        joined = np.vstack((np.hstack((train[0], test[0])), np.hstack((train[1], test[1])))).T
    shuffled_df = pd.DataFrame(data={'Fpath': joined[:,0], 'Labels': joined[:,1]})
    shuffled_df.to_csv('shuffled.csv',index=False)

def main(args):
    params = Params(args.settings)
    ## Data prep
    df = pd.read_csv(params.data, usecols=['Fpath', 'Labels'])
    df = df[~((df.Labels=='unkw') | (df.Labels=='leg'))] # shape: (8785, 2)
    df['Labels'] = df['Labels'].apply(lambda x:x.split(';'))
    X, y = df.Fpath.values, df.Labels.values
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, stratify=y, train_size=0.8, random_state=params.seed)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, stratify=y_tr, train_size=0.9, random_state=params.seed)

    normalize_params = None # {'method': normalize, 'args': [-1000, 1000]}
    if params.aug:
        gen_aug = keras.preprocessing.image.ImageDataGenerator(rotation_range=20, zoom_range=0.20, width_shift_range=0.20, height_shift_range=0.20)
    else: gen_aug = None
    gen_tr = DataGenerator(X_tr, y_tr, batch_size=params.batch_size, n_channels=1, normalize=normalize_params, aug=gen_aug, multi=params.multi)
    gen_val = DataGenerator(X_val, y_val, batch_size=params.batch_size, n_channels=1, normalize=normalize_params, aug=None, multi=params.multi)
    #save_shuffled_df((X_tr,y_tr), val=(X_val,y_val), (X_ts,y_ts))

    if args.train:
        if create_dir(args.train_dir):
            print('continue training') #TODO
        else:
            ## Build model
            if params.model == 'TestModelV3': model = testModelV3(n_class=gen_tr.n_class, shape=gen_tr.data_shape, multi=params.multi).model
            elif params.model == 'NASNetLarge': model = mclass2mlabel(keras.applications.nasnet.NASNetLarge(weights=None, input_shape=gen_tr.data_shape), n_class=gen_tr.n_class)
            elif params.model == 'InceptionResNetV2': model = mclass2mlabel(keras.applications.inception_resnet_v2.InceptionResNetV2(weights=None, input_shape=gen_tr.data_shape), n_class=gen_tr.n_class)
            elif params.model == 'NASNetMobile': model = mclass2mlabel(keras.applications.nasnet.NASNetMobile(weights=None, input_shape=gen_tr.data_shape), n_class=gen_tr.n_class)
            elif params.model == 'MobileNetV2': model = mclass2mlabel(keras.applications.mobilenet_v2.MobileNetV2(weights=None, input_shape=gen_tr.data_shape), n_class=gen_tr.n_class)
            else: print('Unknown model is selected.'); return()
            #elif params.model == 'TestModelV2': model = testModelV2(n_class=gen_tr.n_class, shape=gen_tr.data_shape, multi=params.multi).model
            #elif params.model == 'InceptionV3': model = mclass2mlabel(keras.applications.inception_v3.InceptionV3(weights=None, input_shape=gen_tr.data_shape), n_class=gen_tr.n_class)
            #elif params.model == 'Xception': model = mclass2mlabel(keras.applications.xception.Xception(weights=None, input_shape=gen_tr.data_shape), n_class=gen_tr.n_class)
            #elif params.model == 'ConvNet': model = mclass2mlabel(convNet(gen_tr.data_shape), n_class=gen_tr.n_class)

            optimizer = keras.optimizers.rmsprop(lr=params.lr, decay=params.decay)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            chkpoint = keras.callbacks.ModelCheckpoint(args.train_dir + '/model.h5', monitor='val_loss', save_weights_only=False, save_best_only=True, period=1)
            logger = keras.callbacks.CSVLogger(args.train_dir + '/loss.csv')
            stopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
            model.fit_generator(gen_tr, validation_data=gen_val, epochs=params.n_epochs, verbose=1, use_multiprocessing=False, workers=2, callbacks=[chkpoint, logger,stopper])
            params.save(args.train_dir + '/train_params.json')

    if args.eval:
        model = keras.models.load_model(args.train_dir + '/model.h5')  #model.load_weights(args.weight)
        gen_ts = DataGenerator(X_ts, batch_size=params.batch_size, shuffle=False) #, normalize=normalize_params)
        preds = model.predict_generator(gen_ts, use_multiprocessing=True, verbose=1)
        preds_bool = (preds > args.eval)
        eval = multilabel_confusion_matrix(gen_tr.mlb.transform(y_ts), preds_bool.astype(int))
        #print(eval)
        for i, c in enumerate(gen_tr.classes):
            precision = eval[i,1,1] / (eval[i,1,1] + eval[i,0,1])
            recall = eval[i,1,1] / (eval[i,1,1] + eval[i,1,0])
            print('%s F1 score: %f%%' % (c, 200 * (precision*recall) / (precision + recall)))
            #print('%s accuracy: %f%%' % (c, 100*(eval[i].trace())/eval[i].sum()))

        sw_eval = multilabel_confusion_matrix(gen_tr.mlb.transform(y_ts), preds_bool.astype(int), samplewise=True)
        mask = np.where(sw_eval.trace(axis1=1, axis2=2) != gen_tr.n_class)[0] # number of misclassification
        #X_ts[(sw_eval[:,0,1] != 0) | (sw_eval[:,1,0] != 0)]
        print('Test Accuracy: %f%%' % (100*(len(X_ts)-len(mask))/len(X_ts)))

        prediction = []
        for sample in preds_bool:
            l = [gen_tr.classes[i] for i, v in enumerate(sample) if v]
            if len(l) > 0: prediction.append(','.join(l))
            else: prediction.append('unkw')

        results = pd.DataFrame({'Fpath':X_ts[mask], 'Truth':np.array(y_ts)[mask], 'Predict':np.array(prediction)[mask]})
        results.to_csv("{0}/eval{1:.0f}.csv".format(args.train_dir, args.eval*100), index=False)

    if params.plot_model:
        model.summary()
        keras.utils.plot_model(model, args.train_dir + '/model.png', show_shapes=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('train_dir', nargs='?', default=datetime.today().strftime('%Y%m%d_EXP'))
    parser.add_argument('--settings', default='default_settings.json')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', nargs='?', const=0.95, default=False, type=float)
    main(parser.parse_args())
