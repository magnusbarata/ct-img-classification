#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import glob
import keras
import numpy as np
import pydicom as dcm
import pandas as pd
from generator import DataGenerator
from models import testModel

""" TODO
 - Options: file format, model selector (from json), weight selector
 - Multiple label case
 - Outlier
 - Ordering
"""

CLASSES = ['abd', 'chest', 'head', 'neck', 'pelv']
SORT_ORDER = ['head', 'neck', 'chest', 'abd', 'pelv']
THRESHOLD = 2

def write(fname, X, preds_bool):
    prediction = []
    for sample in preds_bool:
        l = [CLASSES[i] for i, v in enumerate(sample) if v]
        if len(l) > 0: prediction.append(','.join(l))
        else: prediction.append('unkw')

    results = pd.DataFrame({'Fpath':X, 'Predict':prediction})
    results.to_csv(fname, index=False)


def imgPredict(X, model):
    gen = DataGenerator(X, batch_size=8, shuffle=False)
    preds = model.predict_generator(gen, use_multiprocessing=True, verbose=1)
    return (preds > 0.5)

def main(args):
    files = [f for f in glob.glob(args.dir + '**/*.DCM', recursive=True)]
    print('Found %d file(s) from [%s].' %(len(files), args.dir))

    net = testModel(n_class=5, multi=True)
    net.model.load_weights('weights/190805_190726_multi_TestModel_20.h5')
    preds_bool = imgPredict(files, net.model)
    if write is not None: write(args.write, files, preds_bool)

    c_found = [(CLASSES[i], s) for i, s in enumerate(np.sum(preds_bool,axis=0)) if s > THRESHOLD]
    c_found[:] = [y for x in SORT_ORDER for y in c_found if y[0] == x]
    for c in c_found: print('%s : %d slice(s)'%(c[0],c[1]))
    print('%s is a directory consisting CT images from %s to %s.' %(args.dir, c_found[0][0], c_found[-1][0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dir', metavar='CT_images_directory')
    parser.add_argument('-f', '--format', default='.DCM')
    parser.add_argument('-t', '--threshold')
    parser.add_argument('-m', '--model', default='/',
                        help='Keras model which will be used for classification')
    parser.add_argument('-w', '--weight', default='/', help='Model weights')
    parser.add_argument('--write', nargs='?', const='classified.csv', default=None)
    parser.add_argument('--version', action='version', version='%(prog)s 0.2')
    main(parser.parse_args())
