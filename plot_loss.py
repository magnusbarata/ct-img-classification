#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt

with open(sys.argv[1], 'r') as f:
    tkns = []
    for line in f:
        train, val = line.rstrip('\r\n').split(',')
        tkns.append([float(train), float(val)])

data = np.array(tkns)
plt.plot(data[:,0], label='Train')
plt.plot(data[:,1], label='Val')
plt.ylim(0, 1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
