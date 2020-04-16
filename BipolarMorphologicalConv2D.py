# -*- coding: utf8 -*-
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer

class BipolarMorphologicalConv2D(Layer):
    def __init__(self, filters,
                 kernel_size,
                 kernel1_initializer='glorot_uniform',
                 kernel2_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 input_shift=0.1,
                 padding='VALID',
                 strides=(1, 1),
                 **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel1_initializer = kernel1_initializer
        self.kernel2_initializer = kernel2_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.input_shift = input_shift
        self.padding = padding.upper()
        self.strides = strides
        super(BipolarMorphologicalConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel_shape = self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters
        self.k1 = self.add_weight(name='k1',
                                  shape=self.kernel_shape,
                                  initializer=self.kernel1_initializer,
                                  regularizer=self.kernel_regularizer)
        self.k2 = self.add_weight(name='k2',
                                  shape=self.kernel_shape,
                                  initializer=self.kernel2_initializer,
                                  regularizer=self.kernel_regularizer)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer)
        super(BipolarMorphologicalConv2D, self).build(input_shape)

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'kernel1_initializer': self.kernel1_initializer,
                  'kernel2_initializer': self.kernel2_initializer,
                  'bias_initializer': self.bias_initializer,
                  'kernel_regularizer': self.kernel_regularizer,
                  'bias_regularizer': self.bias_regularizer,
                  'input_shift': self.input_shift,
                  'padding': self.padding,
                  'strides': self.strides,}
        base_config = super(BipolarMorphologicalConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        filter_height, filter_width, in_channels, out_channels = self.kernel_shape
        k1_for_patches = tf.reshape(self.k1, [filter_height * filter_width * in_channels, out_channels])
        k2_for_patches = tf.reshape(self.k2, [filter_height * filter_width * in_channels, out_channels])

        x1_patches = tf.extract_image_patches(x, [1, filter_height, filter_width, 1], [1, self.strides[0],
                                             self.strides[1], 1], [1, 1, 1, 1], padding=self.padding)
        x1_patches = tf.maximum(tf.log(tf.maximum(x1_patches, self.input_shift)), tf.float32.min)

        x1_patches = tf.expand_dims(x1_patches, axis=-1)
        y11 = tf.exp(tf.reduce_max(x1_patches + k1_for_patches, 3))
        y12 = tf.exp(tf.reduce_max(x1_patches + k2_for_patches, 3))

        x2_patches = tf.extract_image_patches(-x, [1, filter_height, filter_width, 1], [1, self.strides[0],
                                              self.strides[1], 1], [1, 1, 1, 1], padding=self.padding)
        x2_patches = tf.maximum(tf.log(tf.maximum(x2_patches, self.input_shift)), tf.float32.min)

        x2_patches = tf.expand_dims(x2_patches, axis=-1)
        
        y21 = tf.exp(tf.reduce_max(x2_patches + k1_for_patches, 3))
        y22 = tf.exp(tf.reduce_max(x2_patches + k2_for_patches, 3))
        return tf.nn.bias_add(y11 - y12 - y21 + y22, self.bias)

    def compute_output_shape(self, input_shape):
        if self.padding[0] == 'VALID':
            return (input_shape[0], (input_shape[1] - self.kernel_shape[1] + 1) // self.strides[0],
                   (input_shape[2] - self.kernel_shape[0] + 1) // self.strides[1], self.kernel_shape[3])
        else:
            return (input_shape[0], input_shape[1] // self.strides[0], input_shape[2] // self.strides[1],
                   self.kernel_shape[3])