# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import os
from data_augmentation import rotate,zoom,shift,no_action

def load_npy(item):
    #print(item.numpy())
    data=np.load(item.numpy())
    data=tf.convert_to_tensor(data,dtype=tf.float32)
    return data

def read_image_label(image,label,shape):

    
    image=tf.py_function(load_npy,[image],tf.float32)
    image=image/255.0
    image=(image-0.5)*2
    image.set_shape(shape)
    image=tf.expand_dims(image,axis=-1)
    
    label=tf.py_function(load_npy,[label],tf.float32)
    label=label/255.0
    label=(label-0.5)*2
    label.set_shape(shape)
    label=tf.expand_dims(label,axis=-1)
    return image,label


def read_image_label_seg(source_pair,target,shape):
    source=tf.py_function(load_npy,[source_pair[0]],tf.float32)
    source=source/255.0
    source=(source-0.5)*2
    source.set_shape(shape)
    source=tf.expand_dims(source,axis=-1)
    
    target=tf.py_function(load_npy,[target],tf.float32)
    target=target/255.0
    target=(target-0.5)/2
    target.set_shape(shape)
    target=tf.expand_dims(target,axis=-1)
    
    label=tf.py_function(load_npy,[source_pair[1]],tf.float32)
    label=label/tf.math.reduce_max(label)*9
    label=tf.round(label)
    label=tf.cast(label,tf.uint8)
    label.set_shape(shape)
    label=tf.expand_dims(label,axis=-1)
    return (source,label),target
    
def DataGenerator(cf,file_path,shuffle,aug):
    aug_per=0.25
    ro_range=5
    zoom_range=(0.9,1.1)
    target_size=(100,100,100)
    dx=10;dy=10;dz=10
    
    
    data_files = []
    label_files = []
    fp = open(cf.user_name+file_path)
    lines = fp.readlines()
    fp.close()
    nb_sample = len(lines)
    for line in lines:
        line = line.strip('\n')
        data_files.append(os.path.join(cf.user_name+cf.data_dir,line + '.npy'))
        label_files.append(os.path.join(cf.user_name+cf.label_dir,line + '.npy'))
    ds=tf.data.Dataset.from_tensor_slices((data_files,label_files))
    
    image_ds=ds.map(lambda image,label:image)
    label_ds=ds.map(lambda image,label:label)
    if shuffle:
        image_ds=image_ds.shuffle(buffer_size=nb_sample)
        label_ds=label_ds.shuffle(buffer_size=nb_sample)
    
    ds=tf.data.Dataset.zip((image_ds,label_ds))
    image_label_ds=ds.map(lambda image,label:read_image_label(image,label,cf.shape)) 
    if aug:
        image_label_ds=image_label_ds.map(lambda x,y:tf.cond(tf.random.uniform([], 0, 1) > (1-aug_per),lambda :rotate(x,y,ro_range),lambda:no_action(x,y)))
        image_label_ds=image_label_ds.map(lambda x,y:tf.cond(tf.random.uniform([], 0, 1) > (1-aug_per),lambda :zoom(x,y,zoom_range,target_size),lambda:no_action(x,y)))
        image_label_ds=image_label_ds.map(lambda x,y:tf.cond(tf.random.uniform([], 0, 1) > (1-aug_per),lambda :shift(x,y,dx,dy,dz),lambda:no_action(x,y)))        
    ds=image_label_ds.batch(1)        
    
    return ds

def DataGenerator_seg(cf,file_path,shuffle):
    source_files = []
    target_files = []
    label_files=[]
    fp = open(cf.user_name+file_path)
    lines = fp.readlines()
    fp.close()
    nb_sample = len(lines)
    for line in lines:
        line = line.strip('\n')
        source_files.append(os.path.join(cf.user_name+cf.source_dir,line + '.npy'))
        target_files.append(os.path.join(cf.user_name+cf.target_dir,line + '.npy'))
        label_files.append(os.path.join(cf.user_name+cf.label_dir,line + '.npy'))
    source_ds=tf.data.Dataset.from_tensor_slices((source_files,label_files))
    target_ds=tf.data.Dataset.from_tensor_slices(target_files)
    

    if shuffle:
        source_ds=source_ds.shuffle(buffer_size=nb_sample)
        target_ds=target_ds.shuffle(buffer_size=nb_sample)
    
    ds=tf.data.Dataset.zip((source_ds,target_ds))
    
    image_label_ds=ds.map(lambda source,target:read_image_label_seg(source,target,cf.shape))  
    
    ds=image_label_ds.batch(1)        
    
    return ds