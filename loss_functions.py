# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import backend as K

def mdl(y_true,y_pred):
    smooth=0.00001
    y_pred=K.reshape(y_pred,(-1,K.int_shape(y_pred)[-1]))
    softmax=tf.nn.softmax(y_pred)
    y_true=K.one_hot(tf.cast(K.flatten(y_true),dtype=tf.int32),K.int_shape(y_pred)[-1])
    intersection = K.sum(y_true * softmax,axis=0)
    return 1-K.mean((2. * intersection + smooth) / (K.sum(y_true**2,axis=0) + K.sum(softmax**2,axis=0) + smooth))

def accuracy(y_true,y_pred):
    smooth=0.00000001
    classes=K.int_shape(y_pred)[-1]
    y_pred=tf.argmax(y_pred,axis=-1)
    
    y_pred=tf.cast(y_pred,tf.uint8)
    y_pred=K.one_hot(y_pred,classes)
    unpacked=tf.unstack(y_pred,axis=-1)
    y_pred=tf.stack(unpacked[1:],axis=-1)
    y_true=tf.squeeze(y_true,axis=-1)
    y_true=tf.cast(y_true,tf.uint8)
    
    if classes==10:
        y_true=K.one_hot(y_true,classes)
    elif classes==11:
        y_true=K.one_hot(y_true,classes-1)
        y_WH=tf.to_float(tf.math.reduce_sum(y_true,axis=-1,keepdims=True))
        y_true=tf.concat([y_true,y_WH],axis=-1)
    
    unpacked=tf.unstack(y_true,axis=-1)
    y_true=tf.stack(unpacked[1:],axis=-1)
    #print(y_true)
    #print(y_pred)
    num=2*K.sum(y_true*y_pred,axis=[1,2,3])
    den=K.sum(y_true,axis=[1,2,3])+K.sum(y_pred,axis=[1,2,3])
    return K.mean(num/(den+smooth)) 


    
    
    
    
    