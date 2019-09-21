#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

with open(sys.argv[1], 'r') as f:
    data = [s.strip().split(',') for s in f.readlines()]

out_dir = 'label-based'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    print('Directory based on labeled file:', out_dir)

for tkn in data[1:]:
    label_dir = out_dir + '/' + tkn[-1]
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
        print('Creating directory for [%s] label.' %tkn[-1])

    cp_cmd = 'cp ' + tkn[0] + ' ' + label_dir + '/' + tkn[0].replace('/','%')
    os.system(cp_cmd)
