# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
import numpy as np
from tensorflow.compat.v1.image import resize_bilinear
import tensorflow.keras.backend as K

def residual_block(nb_filters,weight_decay=0.00005,strides=1,d=1,batch_momentum=0.99):
    n1,n2,n3=nb_filters
    def f(input_tensor):
        x=Conv3D(n1,1,strides=strides,kernel_regularizer=l2(weight_decay))(input_tensor)
        x=BatchNormalization(momentum=batch_momentum)(x)
        x=LeakyReLU(alpha=0.1)(x)
        
        x=Conv3D(n2,3,dilation_rate=d,padding='same',kernel_regularizer=l2(weight_decay))(x)
        x=BatchNormalization(momentum=batch_momentum)(x)
        x=LeakyReLU(alpha=0.1)(x)

        x=Conv3D(n3,1,kernel_regularizer=l2(weight_decay))(x)
        x=BatchNormalization(momentum=batch_momentum)(x)
        x=LeakyReLU(alpha=0.1)(x)
        
        x0=Conv3D(n3,1,strides=strides,kernel_regularizer=l2(weight_decay))(input_tensor)
        x0=BatchNormalization(momentum=batch_momentum)(x0)        
        x=Add()([x,x0])
        x=LeakyReLU(alpha=0.1)(x)
        return x
    return f

def conv_block(kernel_size, filters, weight_decay=0., strides=(1, 1, 1), batch_momentum=0.99):
    def f(input_tensor):
        bn_axis = 4
        x = Conv3D(filters, kernel_size, padding='same',kernel_regularizer=l2(weight_decay),strides=strides)(input_tensor)
        x = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        return x
    return f

def attention_gate(n):
    def f(input1,input2):
        xc=Conv3D(n,1,padding='same',strides=2)(input1)
        xp=Conv3D(n,1,padding='same')(input2)
        x=Add()([xc,xp])
        x=ReLU()(x)
        x=UpSampling3D(2)(x)
        x=Conv3D(1,1,padding='same')(x)
        x=tf.math.sigmoid(x)
        
        x=tf.math.multiply(x,input1)
        return x
    return f

def resize_images_bilinear(X, height_factor=1, width_factor=1, depth_factor=1, target_height=None, target_width=None, target_depth=None, data_format='default'):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if data_format == 'default':
        data_format = K.image_data_format()
    if data_format == 'channels_first':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:4]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = tf.transpose(X, [0, 4, 2, 3, 1])
        S = K.int_shape(X)
        X = tf.reshape(X, [S[1], S[2], S[3],S[4]])
        X = resize_bilinear(X, new_shape)
        X = tf.reshape(X, [S[0], S[1], target_height, target_width, S[4]])
        X = tf.transpose(X, [0, 4, 2, 3, 1])

        if target_height and target_width and target_depth:
            X.set_shape((None, None, target_height, target_width, target_depth))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor, original_shape[4]*depth_factor))
        return X
    elif data_format == 'channels_last':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
            new_shape_Z = tf.constant(np.array((target_height, target_depth)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        '''
        #[None,80,80,25,10]
        X = tf.transpose(X, [0, 3, 1, 2, 4]) #[None,25,80,80,10]
        #print('Here*************')
        X = tf.squeeze(X,[0]) #[25,80,80,10]
        X = tf.image.resize_bilinear(X, new_shape)#[25,320,320,10]
        X = tf.transpose(X,[1, 0, 2, 3])#[320,25,320,10]
        X = tf.image.resize_bicubic(X, new_shape_Z)#[320,50,320,10]
        X = tf.transpose(X,[0, 2, 1, 3])#[320,320,50,10]
        X = tf.expand_dims(X,axis=0)#[1,320,320,50,10]
        S = K.int_shape(X)
        #print('*************S=', S)
        X = tf.placeholder_with_default(X,shape=[None,target_height,target_width,target_depth,S[4]])       
        #print('X shape=',X.get_shape())
        '''
        resized_list=[]
        unpack_list=tf.unstack(X,axis=4)
        for i in unpack_list:
            resized_list.append(resize_bilinear(i,new_shape))
        X=tf.stack(resized_list,axis=4)
        X=tf.transpose(X,[0,1,3,2,4]) #[?,320,25,320,10]
        
        resized_list=[]
        unpack_list=tf.unstack(X,axis=4) #[?,320,25,320]
        for i in unpack_list:
            resized_list.append(resize_bilinear(i,new_shape_Z))
        X=tf.stack(resized_list,axis=4) #[?,320,50,320,10]
        X=tf.transpose(X,[0,1,3,2,4])
        
        if target_height and target_width and target_depth:
            X.set_shape((None, target_height, target_width, target_depth, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, original_shape[3] * depth_factor,None))
        #print('X shape is=',K.int_shape(X))
        return X
    else:
        raise Exception('Invalid data_format: ' + data_format)

class BilinearUpSampling3D(Layer):
    def __init__(self, size=(1, 1, 1), target_size=None, data_format='default', **kwargs):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.size = tuple(size)
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpSampling3D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            depth = int(self.size[1] * input_shape[4] if input_shape[4] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
                depth = self.target_size[2]
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height,depth)
        elif self.data_format == 'channels_last':
            width = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)
            depth = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
                depth = self.target_size[2]
            return (input_shape[0],
                    width,
                    height,
                    depth,
                    input_shape[4])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1], target_depth=self.target_size[2], data_format=self.data_format)
        else:
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1],target_depth=self.target_size[2], data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))