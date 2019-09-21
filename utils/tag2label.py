#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from os.path import join, abspath
import sys
import pydicom as dcm

out_file = open(sys.argv[2], 'w')
out_file.write('Fpath,Label\n')
for root, dirs, files in os.walk(sys.argv[1]):
    for i, file in enumerate(files):
        if file.endswith('.DCM'):
            file = abspath(join(root, file))
            print('Opening file (%d/%d): %s' %(i, len(files), file))
            ds = dcm.dcmread(file)
            out_file.write(file + ',' + ds.BodyPartExamined + '\n')

out_file.close()
print('Finished labeling from tag.')
