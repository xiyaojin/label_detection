# -*- coding: utf-8 -*-

import tensorflow as tf
LAMBDA=10

#loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_obj = tf.keras.losses.MeanSquaredError()

def hist_loss(now,hists): #now: tensor, hists:[tensor1,tensor2,...]
    return tf.reduce_mean(tf.square(now-tf.add_n(hists)/len(hists)))

def sum_weights(var):
    weights=tf.constant([],dtype=tf.float32)
    for v in var:
        weights=tf.concat([weights,tf.reshape(v,[-1])],axis=0)
    return weights

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real,dtype='float32')*0.95, real)
      
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
      
    total_disc_loss = real_loss + generated_loss
      
    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image,lmd):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    #loss1 = tf.reduce_mean(tf.math.square(real_image-cycled_image))
    return lmd * loss1


def identity_loss(real_image, same_image, lmd):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    #loss = tf.reduce_mean(tf.math.square(real_image-same_image))
    return lmd * 0.5 * loss

def grad_loss(real,generated):
    real=tf.expand_dims(tf.squeeze(real),0)
    generated=tf.expand_dims(tf.squeeze(generated),0)
    
    sobel_real=tf.image.sobel_edges(real)
    sobel_real1=sobel_real[:,:,:,:,0]
    sobel_real2=sobel_real[:,:,:,:,1]
    
    sobel_generated=tf.image.sobel_edges(generated)
    sobel_generated1=sobel_generated[:,:,:,:,0]
    sobel_generated2=sobel_generated[:,:,:,:,1]
    
    loss = tf.reduce_mean(tf.math.square(sobel_real1-sobel_generated1)+tf.math.square(sobel_real2-sobel_generated2))
    return 0.1*loss

def segmentation_loss(seg,label):
    smooth=0.0000001
    print(label.shape)
    label=tf.one_hot(tf.squeeze(label),10,axis=-1)
    print(label.shape)
    seg=tf.squeeze(tf.nn.softmax(seg,axis=-1))
    print(seg.shape)
    
    label=tf.reshape(label,[-1,10])
    seg=tf.reshape(seg,[-1,10])

    num=2*tf.reduce_sum(tf.math.multiply(label,seg),axis=0)+smooth
    den=tf.math.reduce_sum(label**2,axis=0)+tf.math.reduce_sum(seg**2,axis=0)+smooth
    
    return 10*(1-tf.reduce_mean(num/den))

    
    
    
    # #Cross entropy loss
    # ce=-tf.reduce_sum(tf.math.multiply(label,seg),axis=1)
    # ce=tf.reduce_mean(ce)
    # return 10*ce

    
    
    
    
    