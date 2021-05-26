# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import *
from layers import *
from tensorflow_addons.layers import SpectralNormalization


def generator(filters):


    inputs = tf.keras.layers.Input(shape=[100,100,100,1])
    

    x=SpectralNormalization(Conv3D(64,7,strides=2,padding='same',kernel_regularizer=l2(0.00005)))(inputs)
    x=LeakyReLU(0.1)(x)
    x=residual_block(filters[0],strides=1,d=1)(x)
    x=residual_block(filters[0],strides=1,d=1)(x)
    x1=residual_block(filters[0],strides=1,d=1)(x)
    
    x=residual_block(filters[1],strides=2)(x1)
    x=residual_block(filters[1],strides=1,d=1)(x)
    x2=residual_block(filters[1],strides=1,d=1)(x)
    
    x=residual_block(filters[2],strides=2)(x2)
    x=residual_block(filters[2],strides=1,d=1)(x)
    x=residual_block(filters[2],strides=1,d=1)(x)
    
    x=BilinearUpSampling3D(target_size=(int(100/4), int(100/4), int(100/4), 1))(x)
    x=concatenate([x2,x])
    x=residual_block(filters[1],strides=1,d=1)(x)
    x=residual_block(filters[1],strides=1,d=1)(x)
    x=residual_block(filters[1],strides=1,d=1)(x)
    
    x = BilinearUpSampling3D(target_size=(int(100/2), int(100/2), int(100/2), 1))(x)
    x=concatenate([x1,x])
    x=residual_block(filters[0],strides=1,d=1)(x)        
    x=residual_block(filters[0],strides=1,d=1)(x)   
    x=residual_block(filters[0],strides=1,d=1)(x)
    
    x = BilinearUpSampling3D(target_size=(100,100,100,1))(x)
    x=residual_block(filters[0],strides=1,d=1)(x)        
    x=SpectralNormalization(Conv3D(32,1,padding='same'))(x)
    x=LeakyReLU(0.1)(x)
    x=SpectralNormalization(Conv3D(1,1,padding='same',activation='tanh'))(x)
    
    
    return tf.keras.Model(inputs=inputs,outputs=x)


def discriminator(filters):

    rate=0.05
       
    inputs = tf.keras.layers.Input(shape=[100, 100, 100, 1], name='input_image')
    
    x=GaussianNoise(input_shape=[100,100,100,1],stddev=0.001)(inputs)
    x=SpectralNormalization(Conv3D(64,7,strides=2,padding='same',kernel_regularizer=l2(0.00005)))(inputs)
    x=LeakyReLU(0.1)(x)
    
    x=residual_block(filters[0],strides=1,d=1)(x)
    x=residual_block(filters[0],strides=1,d=1)(x)
    x1=residual_block(filters[0],strides=1,d=1)(x)
    
    x=residual_block(filters[1],strides=2)(x1)
    x=residual_block(filters[1],strides=1,d=1)(x)
    x2=residual_block(filters[1],strides=1,d=1)(x)
    
    x=residual_block(filters[2],strides=2)(x2)
    x=residual_block(filters[2],strides=1,d=1)(x)
    x=residual_block(filters[2],strides=1,d=1)(x)    
 
    
    x=SpectralNormalization(Conv3D(32,1,padding='same'))(x)
    x=SpectralNormalization(Conv3D(1,1,padding='same',activation='sigmoid'))(x)
 
    
    return tf.keras.Model(inputs=inputs, outputs=x)