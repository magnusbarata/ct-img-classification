import keras
from keras.layers import *
from keras.models import Model

def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='elu', dilation_rate=(1, 1), use_bias=False, type='separable', name=None):
    if name is None: bn_name = conv_name = name
    else:
        bn_name = name + '_bn'
        conv_name = name + '_conv'

    if type == 'separable':
        x = SeparableConv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, dilation_rate=dilation_rate, name=conv_name)(x)
    else:
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, dilation_rate=dilation_rate, name=conv_name)(x)
    if not use_bias:
        bn_axis = 1 if keras.backend.image_data_format() == 'channels_first' else 3
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None: x = Activation(activation, name=name)(x)
    return x

def mclass2mlabel(model, n_class=4, flatten=False):
    if flatten: x = Flatten()(model.layers[-2].output)
    else: x = model.layers[-2].output
    out = Dense(n_class, activation='sigmoid')(x)
    return Model(model.input, out, name=model.name+'_multilabel')

class testModelV3:
    """testModelV3."""

    def __init__(self, shape=(512,512,1), n_class=4, multi=False):
        in_img = Input(shape=shape, dtype='float')
        x = conv2d_bn(in_img, 10, 1, type='normal')
        x = conv2d_bn(x, 16, 3, type='normal')
        x = conv2d_bn(x, 16, 3, type='normal')
        x = MaxPooling2D(3, 2)(x)

        x = self.inceptionModuleX(x, 32, 3, 1)
        x = MaxPooling2D(3, 2)(x)
        for i in range(3):
            x = self.inceptionResNetA(x, 64, 1., i)
        x = self.inceptionModuleX(x, 64, 3, 2)
        x = MaxPooling2D(3, 2)(x)

        for i in range(3):
            x = self.inceptionResNetB(x, 64, 7, 1., i)
        x = self.inceptionModuleX(x, 64, 3, 3)
        x = MaxPooling2D(3, 2)(x)

        for i in range(3):
            x = self.inceptionResNetC(x, 128, 1., i)

        x = GlobalAveragePooling2D()(x)
        #x = Flatten()(x)
        if multi: out = Dense(n_class, activation='sigmoid')(x)
        else: out = Dense(n_class, activation='softmax')(x)
        self.model = Model(in_img, out, name='TestModelV3')

    def inceptionResNetA(self, x, n_convs, scale, mod_num, activation='elu'):
        name = 'modA' + str(mod_num)
        branch1 = conv2d_bn(x, n_convs, 1, name=name+'-1x1')

        branch2 = conv2d_bn(x, n_convs, 1, name=name+'br2-1x1')
        branch2 = conv2d_bn(branch2, n_convs, 3, name=name+'br2-3x3_1')
        branch2 = conv2d_bn(branch2, n_convs, 3, name=name+'br2-3x3_2')

        branch3 = conv2d_bn(x, n_convs, 1, name=name+'br3-1x1')
        branch3 = conv2d_bn(branch3, n_convs, 3, name=name+'br3-3x3')

        channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else 3
        concat = Concatenate(axis=channel_axis, name=name+'_cat')([branch1, branch2, branch3])
        residual = conv2d_bn(concat, keras.backend.int_shape(x)[channel_axis], 1, activation=None, use_bias=True, type='normal')
        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                   output_shape=keras.backend.int_shape(x)[1:],
                   arguments={'scale': scale})([x, residual])
        return Activation(activation)(x) if activation is not None else x

    def inceptionResNetB(self, x, n_convs, kernel_size, scale, mod_num, activation='elu'):
        name = 'modB' + str(mod_num)
        branch1 = conv2d_bn(x, n_convs, 1, name=name+'-1x1')

        branch2 = conv2d_bn(x, n_convs, 1, name=name+'br2-1x1')
        branch2 = conv2d_bn(branch2, n_convs, (1, kernel_size), name=name+'br2-1xn_1')
        branch2 = conv2d_bn(branch2, n_convs, (kernel_size, 1), name=name+'br2-nx1_1')
        branch2 = conv2d_bn(branch2, n_convs, (1, kernel_size), name=name+'br2-1xn_2')
        branch2 = conv2d_bn(branch2, n_convs, (kernel_size, 1), name=name+'br2-nx1_2')

        branch3 = conv2d_bn(x, n_convs, 1, name=name+'br3-1x1')
        branch3 = conv2d_bn(branch3, n_convs, (1, kernel_size), name=name+'br3-1xn')
        branch3 = conv2d_bn(branch3, n_convs, (kernel_size, 1), name=name+'br3-nx1')

        channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else 3
        concat = Concatenate(axis=channel_axis, name=name+'_cat')([branch1, branch2, branch3])
        residual = conv2d_bn(concat, keras.backend.int_shape(x)[channel_axis], 1, activation=None, use_bias=True, type='normal')
        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                   output_shape=keras.backend.int_shape(x)[1:],
                   arguments={'scale': scale})([x, residual])
        return Activation(activation)(x) if activation is not None else x

    def inceptionResNetC(self, x, n_convs, scale, mod_num, activation='elu'):
        name = 'modC' + str(mod_num)
        branch1 = conv2d_bn(x, n_convs, 1, name=name+'-1x1')

        branch2 = conv2d_bn(x, n_convs, 1, name=name+'br2-1x1')
        branch2 = conv2d_bn(branch2, n_convs, 3, name=name+'br2-3x3')
        branch2a = conv2d_bn(branch2, n_convs, (1, 3), name=name+'br2-1x3')
        branch2b = conv2d_bn(branch2, n_convs, (3, 1), name=name+'br2-3x1')

        branch3 = conv2d_bn(x, n_convs, 1, name=name+'br3-1x1')
        branch3a = conv2d_bn(branch3, n_convs, (1, 3), name=name+'br3-1x3')
        branch3b = conv2d_bn(branch3, n_convs, (3, 1), name=name+'br3-3x1')

        channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else 3
        concat = Concatenate(axis=channel_axis, name=name+'_cat')([branch1, branch2a, branch2b, branch3a, branch3b])
        residual = conv2d_bn(concat, keras.backend.int_shape(x)[channel_axis], 1, activation=None, use_bias=True, type='normal')
        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                   output_shape=keras.backend.int_shape(x)[1:],
                   arguments={'scale': scale})([x, residual])
        return Activation(activation)(x) if activation is not None else x

    def inceptionModuleX(self, x, n_convs, kernel_size, mod_num):
        name = 'modX' + str(mod_num)
        branch1 = conv2d_bn(x, n_convs, 3, dilation_rate=6, name=name+'_r6')
        branch2 = conv2d_bn(x, n_convs, 3, dilation_rate=12, name=name+'_r12')
        branch3 = conv2d_bn(x, n_convs, 3, dilation_rate=18, name=name+'_r18')
        branch4 = conv2d_bn(x, n_convs, 3, dilation_rate=24, name=name+'_r24')

        return Concatenate(name=name+'_cat')([branch1, branch2, branch3, branch4])
