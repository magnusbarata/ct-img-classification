#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras
from keras.layers import Input, Conv2D, Dense, Concatenate, Flatten, Add,\
                         MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import plot_model

class testModel:
    def __init__(self, shape=(512, 512, 1), n_class=4, multi=False):
        in_img = Input(shape=(512, 512, 1), dtype='float')
        conv = Conv2D(16, 3, activation='relu')(in_img)
        conv = Conv2D(16, 3, activation='relu')(conv)
        conv = MaxPooling2D(3, 2)(conv)

        conv = self.inceptionModuleX(conv, 32, 3, 1)
        conv = MaxPooling2D(3, 2)(conv)
        for i in range(5):
            conv = self.inceptionModuleA(conv, 64, i)
        conv = MaxPooling2D(3, 2)(conv)

        for i in range(3):
            conv = self.inceptionModuleB(conv, 64, 7, i)
        conv = MaxPooling2D(3, 2)(conv)

        conv = self.inceptionModuleC(conv, 64, 1)
        conv = self.inceptionModuleC(conv, 64, 2)
        #GAP = GlobalAveragePooling2D()(conv)
        #FC = Dense(128, activation='relu')(GAP)
        flat = Flatten()(conv)
        if multi: out = Dense(n_class, activation='sigmoid')(flat)
        else: out = Dense(n_class, activation='softmax')(flat)
        self.model = Model(in_img, out)

    def inceptionModuleA(self, conv, n_convs, mod_num):
        name = 'modA' + str(mod_num)
        branch1 = Conv2D(n_convs, 1, activation='relu', name=name+'-1x1')(conv)

        branch2 = Conv2D(n_convs, 1, activation='relu', name=name+'br2-1x1')(conv)
        branch2 = Conv2D(n_convs, 3, activation='relu', padding='same', name=name+'br2-3x3_1')(branch2)
        branch2 = Conv2D(n_convs, 3, activation='relu', padding='same', name=name+'br2-3x3_2')(branch2)

        branch3 = Conv2D(n_convs, 1, activation='relu', name=name+'br3-1x1')(conv)
        branch3 = Conv2D(n_convs, 3, activation='relu', padding='same', name=name+'br3-3x3')(branch3)

        branch4 = AveragePooling2D(2, 1, name=name+'br4-AvgPool', padding='same')(conv)
        branch4 = Conv2D(n_convs, 1, activation='relu', name=name+'br4-1x1')(branch4)
        return Concatenate(name=name+'_cat')([branch1, branch2, branch3, branch4])

    def inceptionModuleB(self, conv, n_convs, kernel_size, mod_num):
        name = 'modB' + str(mod_num)
        branch1 = Conv2D(n_convs, 1, activation='relu', name=name+'-1x1')(conv)

        branch2 = Conv2D(n_convs, 1, activation='relu', name=name+'br2-1x1')(conv)
        branch2 = Conv2D(n_convs, (1, kernel_size), activation='relu', padding='same', name=name+'br2-1xn_1')(branch2)
        branch2 = Conv2D(n_convs, (kernel_size, 1), activation='relu', padding='same', name=name+'br2-nx1_1')(branch2)
        branch2 = Conv2D(n_convs, (1, kernel_size), activation='relu', padding='same', name=name+'br2-1xn_2')(branch2)
        branch2 = Conv2D(n_convs, (kernel_size, 1), activation='relu', padding='same', name=name+'br2-nx1_2')(branch2)

        branch3 = Conv2D(n_convs, 1, activation='relu', name=name+'br3-1x1')(conv)
        branch3 = Conv2D(n_convs, (1, kernel_size), activation='relu', padding='same', name=name+'br3-1xn')(branch3)
        branch3 = Conv2D(n_convs, (kernel_size, 1), activation='relu', padding='same', name=name+'br3-nx1')(branch3)

        branch4 = AveragePooling2D(2, 1, name=name+'br4-AvgPool', padding='same')(conv)
        branch4 = Conv2D(n_convs, 1, activation='relu', name=name+'br4-1x1')(branch4)
        return Concatenate(name=name+'_cat')([branch1, branch2, branch3, branch4])

    def inceptionModuleC(self, conv, n_convs, mod_num):
        name = 'modC' + str(mod_num)
        branch1 = Conv2D(n_convs, 1, activation='relu', name=name+'-1x1')(conv)

        branch2 = Conv2D(n_convs, 1, activation='relu', name=name+'br2-1x1')(conv)
        branch2 = Conv2D(n_convs, 3, activation='relu', padding='same', name=name+'br2-3x3')(branch2)
        branch2a = Conv2D(n_convs, (1, 3), activation='relu', padding='same', name=name+'br2-1x3')(branch2)
        branch2b = Conv2D(n_convs, (3, 1), activation='relu', padding='same', name=name+'br2-3x1')(branch2)

        branch3 = Conv2D(n_convs, 1, activation='relu', name=name+'br3-1x1')(conv)
        branch3a = Conv2D(n_convs, (1, 3), activation='relu', padding='same', name=name+'br3-1x3')(branch3)
        branch3b = Conv2D(n_convs, (3, 1), activation='relu', padding='same', name=name+'br3-3x1')(branch3)

        branch4 = AveragePooling2D(2, 1, name=name+'br4-AvgPool', padding='same')(conv)
        branch4 = Conv2D(n_convs, 1, activation='relu', name=name+'br4-1x1')(branch4)
        return Concatenate(name=name+'_cat')([branch1, branch2a, branch2b, branch3a, branch3b, branch4])

    def inceptionModuleX(self, conv, n_convs, kernel_size, mod_num):
        name = 'modX' + str(mod_num)
        branch1 = Conv2D(n_convs, 3, activation='relu', dilation_rate=6, padding='same', name=name+'_r6')(conv)
        branch2 = Conv2D(n_convs, 3, activation='relu', dilation_rate=12, padding='same', name=name+'_r12')(conv)
        branch3 = Conv2D(n_convs, 3, activation='relu', dilation_rate=18, padding='same', name=name+'_r18')(conv)
        branch4 = Conv2D(n_convs, 3, activation='relu', dilation_rate=24, padding='same', name=name+'_r24')(conv)

        return Concatenate(name=name+'_cat')([branch1, branch2, branch3, branch4])


"""net = testModel()
model = net.model
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
plot_model(model, to_file='model.png')"""
