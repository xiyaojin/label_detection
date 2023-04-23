# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import*
from layers import residual_block,conv_block,BilinearUpSampling3D
from tensorflow.keras.initializers import glorot_normal
 

def Res():
    
    inputs = Input(shape=[100,100,100,1])
    x=Conv3D(32,7,padding='same',kernel_initializer=glorot_normal())(inputs)
    x=residual_block([8,8,32],strides=1)(x)
    #x=residual_block([4,4,8],strides=1)(x)
    
    x=residual_block([4,4,16],strides=2)(x)
    #x=residual_block([8,8,16],strides=1)(x)
    
    x=residual_block([2,2,8],strides=2)(x)
    #x=residual_block([16,16,32],strides=1)(x)
    
    x=UpSampling3D(2)(x)
    x=residual_block([4,4,16],strides=1)(x)
    #x=residual_block([8,8,32],strides=1)(x)
    
    x=UpSampling3D(2)(x)
    x=residual_block([8,8,32],strides=1)(x)        
    #x=residual_block([16,16,64],strides=1)(x)   
    x=Conv3D(32,1,padding='same',activation='relu')(x)
    x=Conv3D(10,1,padding='same')(x)
       
    return tf.keras.Model(inputs=inputs,outputs=x)   
    

def AE():
    
    img_shape=(100,100,100,1)
    batch_momentum=0.9
    bn_axis=4
    weight_decay=0.00005
    
    input_image=Input(shape=img_shape,name='in_image')
    x1=Conv3D(32,7,padding='same',kernel_initializer=glorot_normal())(input_image)
    x1=BatchNormalization(axis=bn_axis,momentum=batch_momentum)(x1)
    x1=Activation('relu')(x1)
    
    x1=conv_block(3, 8, strides=2, weight_decay=weight_decay, batch_momentum=0.95)(x1)
    x1=conv_block(3, 8, strides=1, weight_decay=weight_decay, batch_momentum=0.95)(x1)
    
    x1=conv_block(3, 16, strides=2, weight_decay=weight_decay, batch_momentum=0.95)(x1)
    x1=conv_block(3, 16, strides=1, weight_decay=weight_decay, batch_momentum=0.95)(x1)
    
    x1=conv_block(3, 32, strides=2, weight_decay=weight_decay, batch_momentum=0.95)(x1)
    x1=conv_block(3, 32, strides=1, weight_decay=weight_decay, batch_momentum=0.95)(x1)

    x1=BilinearUpSampling3D(target_size=(25, 25, 25, 1))(x1)
    x1=conv_block(3, 16, weight_decay=weight_decay, batch_momentum=0.95)(x1)
    x1=conv_block(3, 16, weight_decay=weight_decay, batch_momentum=0.95)(x1)
    
    x1=BilinearUpSampling3D(target_size=(50, 50, 50, 1))(x1)
    x1=conv_block(3, 8, weight_decay=weight_decay, batch_momentum=0.95)(x1)
    x1=conv_block(3, 8, weight_decay=weight_decay, batch_momentum=0.95)(x1)
    
    x1=BilinearUpSampling3D(target_size=(100, 100, 100, 1))(x1)
    x1=conv_block(3, 4, weight_decay=weight_decay, batch_momentum=0.95)(x1)
    x1=conv_block(3, 4, weight_decay=weight_decay, batch_momentum=0.95)(x1)
    x1=Conv3D(10,1,padding='same',kernel_initializer=glorot_normal())(x1)
    x1=Activation('linear')(x1)
    
    return tf.keras.Model(inputs=input_image,outputs=x1)

