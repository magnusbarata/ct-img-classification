#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time, datetime
import argparse
import pandas as pd
import numpy as np
import keras
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from generator import DataGenerator
from models import testModel, testModelV2
from models import mclass2mlabel
from keras.utils import plot_model

def save_shuffled_df(train, val, test):
    if val is not None:
        joined = np.vstack((np.hstack((train[0], val[0], test[0])), np.hstack((train[1], val[1], test[1])))).T
    else:
        joined = np.vstack((np.hstack((train[0], test[0])), np.hstack((train[1], test[1])))).T
    shuffled_df = pd.DataFrame(data={'Fpath': joined[:,0], 'Labels': joined[:,1]})
    shuffled_df.to_csv('shuffled.csv',index=False)

def main(args):
    ## Data prep
    seed = args.seed
    df = pd.read_csv(args.data, usecols=['Fpath', 'Labels'])
    df = df[~((df.Labels=='unkw') | (df.Labels=='leg'))] # shape: (8785, 2)
    df['Labels'] = df['Labels'].apply(lambda x:x.split(';'))
    X, y = df.Fpath.values, df.Labels.values
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, stratify=y, train_size=0.8, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, stratify=y_tr, train_size=0.8, random_state=seed)

    gen_params = {'dim': (512,512), 'batch_size': args.batch, 'shuffle': True,
                  'n_channels':1, 'multi': True}
    gen_tr = DataGenerator(X_tr, y_tr, **gen_params)
    gen_val = DataGenerator(X_val, y_val, **gen_params)
    #save_shuffled_df((X_tr,y_tr), (X_val,y_val), (X_ts,y_ts))

    ## Build model
    if args.model == 'TestModel': model = testModel(n_class=gen_tr.n_class, multi=True).model
    elif args.model == 'TestModelV2': model = testModelV2(n_class=gen_tr.n_class, multi=True).model
    elif args.model == 'InceptionV3': model = mclass2mlabel(keras.applications.inception_v3.InceptionV3(weights=None, input_shape=(512,512,1)), n_class=gen_tr.n_class)
    elif args.model == 'Xception': model = mclass2mlabel(keras.applications.xception.Xception(weights=None, input_shape=(512,512,1)), n_class=gen_tr.n_class)
    elif args.model == 'MobileNetV2': model = mclass2mlabel(keras.applications.mobilenet_v2.MobileNetV2(weights=None, input_shape=(512,512,1)), n_class=gen_tr.n_class)
    elif args.model == 'InceptionResNetV2': model = mclass2mlabel(keras.applications.inception_resnet_v2.InceptionResNetV2(weights=None, input_shape=(512,512,1)), n_class=gen_tr.n_class)
    else: print('Unknown model is selected.'); return()
    optimizer = optimizers.rmsprop(lr=1e-5, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #model.summary()
    #plot_model(model, to_file=model.name+'.png')

    if args.train:
        train_date = ''.join(str(datetime.date.today())[2:].split('-'))
        weight_f = 'weights/' + train_date + '_' + args.data[-10:-4] + '_multi_' + model.name + '.h5'
        chkpoint = keras.callbacks.ModelCheckpoint(weight_f, monitor='val_loss', save_weights_only=False, save_best_only=True, period=1)
        logger = keras.callbacks.CSVLogger('loss-files/' + weight_f[8:-3] + '-loss_file.csv')
        start_time = time.time()
        model.fit_generator(gen_tr, validation_data=gen_val, epochs=100, verbose=1, use_multiprocessing=True, workers=4, callbacks=[chkpoint, logger])
        print('Elapsed time: ', datetime.timedelta(seconds=time.time()-start_time))

    if args.eval:
        model.load_weights(args.weight)
        gen_ts = DataGenerator(X_ts, batch_size=args.batch, shuffle=False)
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
        #results.to_csv('results.csv',index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', default=42)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--data', default='datasets/train_data_190726.csv')
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--model', default='TestModel')
    parser.add_argument('--weight', default='weights/190805_190726_multi_TestModel_20.h5')
    main(parser.parse_args())
