# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
#tf.enable_eager_execution()
from tensorflow.compat.v1.image import resize_bilinear,resize_nearest_neighbor
def rotate(x,y,ro_range):

    pi=tf.constant(np.pi)
    x1=x


    ro_range=ro_range/180*pi
    #tf.set_random_seed(1)
    angle_a=tf.random.uniform(shape=[],minval=-ro_range,maxval=ro_range,dtype=tf.float32)
    angle_b=tf.random.uniform(shape=[],minval=-ro_range,maxval=ro_range,dtype=tf.float32)
    angle_c=tf.random.uniform(shape=[],minval=-ro_range,maxval=ro_range,dtype=tf.float32)
    x1=tf.transpose(x1,[3,0,1,2]) #(1,320,320,50)
    x1=tfa.image.rotate(x1,angle_a,interpolation='NEAREST')
    x1=tf.transpose(x1,[0,1,3,2]) #(1,320,50,320)
    x1=tfa.image.rotate(x1,angle_b,interpolation='NEAREST')
    x1=tf.transpose(x1,[0,3,2,1]) #(1,320,50,320)
    x1=tfa.image.rotate(x1,angle_c,interpolation='NEAREST')
    x1=tf.transpose(x1,[3,1,2,0]) #(320,320,50,1)

    # x2=tf.transpose(x2,[3,0,1,2]) #(1,320,320,50)
    # x2=tfa.image.rotate(x2,angle_a,interpolation='NEAREST')
    # x2=tf.transpose(x2,[0,1,3,2]) #(1,320,50,320)
    # x2=tfa.image.rotate(x2,angle_b,interpolation='NEAREST')
    # x2=tf.transpose(x2,[0,3,2,1]) #(1,320,50,320)
    # x2=tfa.image.rotate(x2,angle_c,interpolation='NEAREST')
    # x2=tf.transpose(x2,[3,1,2,0]) #(320,320,50,1)
    
    angle_a=tf.random.uniform(shape=[],minval=-ro_range,maxval=ro_range,dtype=tf.float32)
    angle_b=tf.random.uniform(shape=[],minval=-ro_range,maxval=ro_range,dtype=tf.float32)
    angle_c=tf.random.uniform(shape=[],minval=-ro_range,maxval=ro_range,dtype=tf.float32)        
    y=tf.transpose(y,[3,0,1,2])
    y=tfa.image.rotate(y,angle_a,interpolation='NEAREST')
    y=tf.transpose(y,[0,1,3,2])
    y=tfa.image.rotate(y,angle_b,interpolation='NEAREST')
    y=tf.transpose(y,[0,3,2,1])
    y=tfa.image.rotate(y,angle_c,interpolation='NEAREST')
    y=tf.transpose(y,[3,1,2,0])    
        
    return x1,y

def zoom(x,y,zoom_range,target_size):
#    tf.enable_eager_execution()
    Minval,Maxval=zoom_range
    print ('minval=',Minval)
    #tf.set_random_seed(1)
    rate_a=tf.random.uniform(shape=[],minval=Minval,maxval=Maxval,dtype=tf.float32)
    rate_b=tf.random.uniform(shape=[],minval=Minval,maxval=Maxval,dtype=tf.float32)
    rate_c=tf.random.uniform(shape=[],minval=Minval,maxval=Maxval,dtype=tf.float32)
    
    target_height,target_width,target_depth=target_size
    x1=x

    x1=tf.transpose(x1,[3,0,1,2])
    x1=resize_bilinear(x1,tf.cast((target_height*rate_a,target_width*rate_a),dtype=tf.int32))
    x1=tf.image.resize_with_crop_or_pad(x1,target_height,target_width)
    x1=tf.transpose(x1,[0,1,3,2])
    x1=resize_bilinear(x1,tf.cast((target_height*rate_b,target_depth*rate_b),dtype=tf.int32))
    x1=tf.image.resize_with_crop_or_pad(x1,target_height,target_depth)
    x1=tf.transpose(x1,[0,3,2,1])
    x1=resize_bilinear(x1,tf.cast((target_width*rate_c,target_depth*rate_c),dtype=tf.int32))
    x1=tf.image.resize_with_crop_or_pad(x1,target_width,target_depth)
    x1=tf.transpose(x1,[3,1,2,0])

    # x=tf.transpose(x,[3,0,1,2])
    # x2=resize_bilinear(x2,tf.cast((target_height*rate_a,target_width*rate_a),dtype=tf.int32))
    # x2=tf.image.resize_with_crop_or_pad(x2,target_height,target_width)
    # x2=tf.transpose(x2,[0,1,3,2])
    # x2=resize_bilinear(x2,tf.cast((target_height*rate_b,target_depth*rate_b),dtype=tf.int32))
    # x2=tf.image.resize_with_crop_or_pad(x2,target_height,target_depth)
    # x2=tf.transpose(x2,[0,3,2,1])
    # x2=resize_bilinear(x2,tf.cast((target_width*rate_c,target_depth*rate_c),dtype=tf.int32))
    # x2=tf.image.resize_with_crop_or_pad(x2,target_width,target_depth)
    # x2=tf.transpose(x2,[3,1,2,0])

    rate_a=tf.random.uniform(shape=[],minval=Minval,maxval=Maxval,dtype=tf.float32)
    rate_b=tf.random.uniform(shape=[],minval=Minval,maxval=Maxval,dtype=tf.float32)
    rate_c=tf.random.uniform(shape=[],minval=Minval,maxval=Maxval,dtype=tf.float32)
    
    y=tf.transpose(y,[3,0,1,2])
    y=resize_nearest_neighbor(y,tf.cast((target_height*rate_a,target_width*rate_a),dtype=tf.int32))
    y=tf.image.resize_with_crop_or_pad(y,target_height,target_width)
    y=tf.transpose(y,[0,1,3,2])
    y=resize_nearest_neighbor(y,tf.cast((target_height*rate_b,target_depth*rate_b),dtype=tf.int32))
    y=tf.image.resize_with_crop_or_pad(y,target_height,target_depth)
    y=tf.transpose(y,[0,3,2,1])
    y=resize_nearest_neighbor(y,tf.cast((target_width*rate_c,target_depth*rate_c),dtype=tf.int32))
    y=tf.image.resize_with_crop_or_pad(y,target_width,target_depth)
    y=tf.transpose(y,[3,1,2,0])   
    y=tf.cast(y,tf.float32)
    
    return x1,y

def shift(x,y,dx_range,dy_range,dz_range):
    #tf.set_random_seed(1)
    dx=tf.random.uniform(shape=[],minval=-dx_range,maxval=dx_range,dtype=tf.int32)
    dy=tf.random.uniform(shape=[],minval=-dy_range,maxval=dy_range,dtype=tf.int32)
    dz=tf.random.uniform(shape=[],minval=-dz_range,maxval=dz_range,dtype=tf.int32)
    
    x1=x
    
    x1=tf.transpose(x1,[3,0,1,2])
    x1=tfa.image.translate(x1,[dx,dy])
    x1=tf.transpose(x1,[0,1,3,2])
    x1=tfa.image.translate(x1,[dx,dz])
    x1=tf.transpose(x1,[0,3,2,1])
    x1=tfa.image.translate(x1,[dy,dz])
    x1=tf.transpose(x1,[3,1,2,0])
    
    # x2=tf.transpose(x2,[3,0,1,2])
    # x2=tfa.image.translate(x2,[dx,dy])
    # x2=tf.transpose(x2,[0,1,3,2])
    # x2=tfa.image.translate(x2,[dx,dz])
    # x2=tf.transpose(x2,[0,3,2,1])
    # x2=tfa.image.translate(x2,[dy,dz])
    # x2=tf.transpose(x2,[3,1,2,0])

    dx=tf.random.uniform(shape=[],minval=-dx_range,maxval=dx_range,dtype=tf.int32)
    dy=tf.random.uniform(shape=[],minval=-dy_range,maxval=dy_range,dtype=tf.int32)
    dz=tf.random.uniform(shape=[],minval=-dz_range,maxval=dz_range,dtype=tf.int32)
    
    y=tf.transpose(y,[3,0,1,2])
    y=tfa.image.translate(y,[dx,dy])
    y=tf.transpose(y,[0,1,3,2])
    y=tfa.image.translate(y,[dx,dz])
    y=tf.transpose(y,[0,3,2,1])
    y=tfa.image.translate(y,[dy,dz])
    y=tf.transpose(y,[3,1,2,0])    
    
    return x1,y

def no_action(x,y):
    return x,y
