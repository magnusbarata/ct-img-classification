#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import pydicom as dcm
import pandas as pd

def main(args):
    df = pd.read_csv(args.label, usecols=args.cols)
    for f, l in zip(df[args.cols[0]], df[args.cols[-1]]):
        ds = dcm.dcmread(f)
        ds.ManufacturerModelName = l

        if args.dir is None: save_path = 'LABELED_' + os.path.dirname(f)
        else: save_path = os.path.join(args.dir, os.path.dirname(f))

        if not os.path.exists(save_path):
            print('labelling files on [%s]...' % save_path)
            os.makedirs(save_path)
        ds.save_as(os.path.join(save_path, os.path.basename(f)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('label', metavar='csv_label_file')
    parser.add_argument('-d', '--dir', metavar='labeled_CT_images_directory')
    parser.add_argument('-c', '--cols', nargs=2, default=['Fpath', 'Labels'])
    main(parser.parse_args())
