# -*- coding: utf-8 -*-
from comet_ml import Experiment
import tensorflow as tf
import shutil
import time
from models import generator,discriminator
from data_generator import DataGenerator
from loss_functions import *
import SimpleITK as sitk
import imp
import os

import numpy as np
from argparse import ArgumentParser
from process_config import process_config
parser=ArgumentParser(description='Model training')

parser.add_argument('-c', '--config_file', help='experiment config file')
args = parser.parse_args()
assert args.config_file is not None, 'Please provide an experiment configuration'\
                                    'file using -c /path/to/config.py'\
                                    ' in the command line'

config_path=args.config_file
cf=imp.load_source('config',config_path)
cf=process_config(cf)

shutil.copyfile(config_path, os.path.join(cf.save_path, "config.py"))
EPOCHS=cf.epochs

experiment=Experiment(api_key='YmSX7HeUw7Lcx3KN18iQe47yi',project_name='cycleGAN',workspace='xiyaojin')
experiment.set_name(cf.experiment_name)



ds_train=DataGenerator(cf,cf.file_path_train,shuffle=True,aug=True)
ds_test=DataGenerator(cf,cf.file_path_test,shuffle=False,aug=False)

fp = open(cf.user_name+cf.file_path_train)
lines = fp.readlines()
N=len(lines)
fp.close()   


generator_g=generator(cf.filters_gen)
generator_f=generator(cf.filters_gen)
discriminator_x=discriminator(cf.filters_disc)
discriminator_y=discriminator(cf.filters_disc)
buffer=2
hist_gen_g=[]
hist_gen_f=[]
hist_disc_x=[]
hist_disc_y=[]

#lr_gen=tf.keras.optimizers.schedules.ExponentialDecay(1e-4,decay_steps=1000,decay_rate=0.5)
#lr_disc=tf.keras.optimizers.schedules.ExponentialDecay(5e-5,decay_steps=1000,decay_rate=0.5)

lr_gen=cf.lr_gen
lr_disc=cf.lr_disc

generator_g_optimizer = tf.keras.optimizers.Adam(lr_gen, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(lr_gen, beta_1=0.5)
discriminator_x_optimizer = tf.keras.optimizers.Adam(lr_disc, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(lr_disc, beta_1=0.5)

checkpoint_path = cf.save_path

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.

starting_epoch=0
if cf.resume==True:
    starting_epoch=cf.starting_epoch
    status=ckpt.restore(os.path.join(cf.save_path,cf.ckpt_file))
    #status=ckpt.restore(ckpt_manager.latest_checkpoint)
    status.assert_existing_objects_matched()

    print ('Latest checkpoint restored!!')

def generate_images_from_patch(gen,ds_test,cf,epoch):
    whole=np.zeros((200,200,200))
    coeff=np.zeros((200,200,200))
    c=0
    for x,y in ds_test:        
        c+=1
        dep=(c-1)//9; row=((c-dep*9)-1)//3; col=((c-dep*9)-1)%3
        pred=gen(x)
        pred=np.squeeze(pred)
        pred=np.transpose(pred,[2,0,1])
        whole[dep*60:dep*60+80,row*60:row*60+80,col*60:col*60+80]+=pred
        coeff[dep*60:dep*60+80,row*60:row*60+80,col*60:col*60+80]+=np.ones((80,80,80))
    whole=whole/coeff
    whole=(whole+1)/2.0
    whole=np.round(whole*255)
    whole[whole<0]=0   
    whole=sitk.Cast(sitk.GetImageFromArray(whole),sitk.sitkUInt8)
    sitk.WriteImage(whole,os.path.join(cf.save_path,'test'+str(epoch+1)+'.nii.gz'))  
    
def generate_images(gen,test_input,cf,epoch,phase,name=''):
    pred=gen(test_input)
    pred=np.squeeze(pred)
    pred=np.transpose(pred,[2,0,1])
    pred=(pred+1)/2.0
    pred=np.round(pred*255)
    pred[pred<0]=0
    pred_sitk=sitk.GetImageFromArray(pred)
    pred_sitk=sitk.Cast(pred_sitk,sitk.sitkUInt8)   

    if phase=='pred':
        save_path=os.path.join(cf.save_path,'generated_images')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sitk.WriteImage(pred_sitk,os.path.join(save_path,name+'.nii.gz'))
    else:
        sitk.WriteImage(pred_sitk,os.path.join(cf.save_path,phase+'_'+str(epoch+1)+'.nii.gz'))

global ratio
ratio=cf.step_ratio



@tf.function
def train_step_gen(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)
    
        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)
    
        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)
    
        disc_real_x = discriminator_x(real_x,training=True)
        disc_real_y = discriminator_y(real_y,training=True)
    
        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)
    
        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)
        
        grad_g_loss = grad_loss(real_x,fake_y)
        grad_f_loss = grad_loss(real_y,fake_x)
        
        total_cycle_loss = calc_cycle_loss(real_x, cycled_x, lmd) + calc_cycle_loss(real_y, cycled_y,lmd)
        
        identity_loss_y=identity_loss(real_y, same_y, lmd)
        identity_loss_x=identity_loss(real_x, same_x, lmd)
        
        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + grad_g_loss + identity_loss_y
        total_gen_f_loss = gen_f_loss + total_cycle_loss + grad_f_loss + identity_loss_x 
    
        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
        
  # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                          generator_f.trainable_variables)   
    # Apply the gradients to the optimizer

    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                              generator_g.trainable_variables))
      
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                              generator_f.trainable_variables))

    return total_gen_g_loss,total_gen_f_loss,disc_x_loss,disc_y_loss,gen_g_loss,total_cycle_loss,identity_loss_y, \
         generator_g.trainable_variables

@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.

    with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.       
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)
    
        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)
    
        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)
    
        disc_real_x = discriminator_x(real_x,training=True)
        disc_real_y = discriminator_y(real_y,training=True)
    
        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)
    
        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)
        
        grad_g_loss = grad_loss(real_x,fake_y)
        grad_f_loss = grad_loss(real_y,fake_x)
        
        total_cycle_loss = calc_cycle_loss(real_x, cycled_x, lmd) + calc_cycle_loss(real_y, cycled_y,lmd)
        
        identity_loss_y=identity_loss(real_y, same_y, lmd)
        identity_loss_x=identity_loss(real_x, same_x, lmd)
        
        # hist_loss_g=hist_loss(sum_weights(generator_g.trainable_variables),hist_gen_g)
        # hist_loss_f=hist_loss(sum_weights(generator_f.trainable_variables),hist_gen_f)
        # hist_loss_x=hist_loss(sum_weights(discriminator_x.trainable_variables),hist_disc_x)
        # hist_loss_y=hist_loss(sum_weights(discriminator_y.trainable_variables),hist_disc_y)
        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + grad_g_loss + identity_loss_y #+ hist_loss_g
        total_gen_f_loss = gen_f_loss + total_cycle_loss + grad_f_loss + identity_loss_x #+ hist_loss_f
    
        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x) #+ hist_loss_x
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y) #+ hist_loss_y
        
  # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                          generator_f.trainable_variables)
        
    discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                          discriminator_x.trainable_variables)        
    discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                  discriminator_y.trainable_variables) 
    
    # Apply the gradients to the optimizer

    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                              generator_g.trainable_variables))     
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                              generator_f.trainable_variables))
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                              discriminator_x.trainable_variables))           
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))    
    
    return total_gen_g_loss,total_gen_f_loss,disc_x_loss,disc_y_loss,gen_g_loss,total_cycle_loss,identity_loss_y
            
  
k=0



for i in range(EPOCHS):
    epoch=i+starting_epoch
    start = time.time()
    print('epoch:'+str(epoch+1))
    global lmd
    lmd=0

    n = 0
    for image_x, image_y in ds_train:
        #print (tf.shape(image_x),tf.shape(image_y))
        k+=1
        
        if k==ratio:

            total_gen_g,total_gen_f,disc_x,disc_y,gen_g,cycle,identity=train_step(image_x, image_y)
            k=0
        else:
   
            total_gen_g,total_gen_f,disc_x,disc_y,gen_g,cycle,identity,var_g=train_step_gen(image_x, image_y)
        print(str(n+1)+'/24,','gen_g_loss:',gen_g.numpy(),
                              'disc_y_loss:',disc_y.numpy(),
              end='\r')
        n+=1

        experiment.log_metric('total_gen_g_loss',total_gen_g.numpy(),step=epoch*N+n)
        experiment.log_metric('total_gen_f_loss',total_gen_f.numpy(),step=epoch*N+n)
        experiment.log_metric('gen_g_loss',gen_g.numpy(),step=epoch*N+n)
        experiment.log_metric('cycle_loss',cycle.numpy(),step=epoch*N+n)
        experiment.log_metric('identity_loss',identity.numpy(),step=epoch*N+n)
        experiment.log_metric('disc_x_loss',disc_x.numpy(),step=epoch*N+n)
        experiment.log_metric('disc_y_loss',disc_y.numpy(),step=epoch*N+n)
        experiment.log_metric('epoch',epoch+1,step=epoch*N+n)

    # Using a consistent image (sample_horse) so that the progress of the model
    # is clearly visible.
    print('\n')
    if (epoch + 1) % 20 == 0:
        ckpt_save_path = ckpt.save(cf.save_path+'/ckpt')
        idx=(epoch+1)//20
        to_delete=os.path.join(cf.save_path,'ckpt-'+str(idx-1)+'.data-00000-of-00001')
        to_delete2=os.path.join(cf.save_path,'ckpt-'+str(idx-1)+'.index')
        if os.path.exists(to_delete):
            os.remove(to_delete)
            os.remove(to_delete2)
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                           ckpt_save_path))
    if (epoch + 1) % 25 == 0:
        train_sample=next(iter(ds_train))[0]
        generate_images(generator_g,train_sample,cf,epoch,'train')
        test_sample=next(iter(ds_test))[0]
        generate_images(generator_g,test_sample,cf,epoch,'test')        
        #generate_images_from_patch(generator_g,ds_test,cf,epoch)
    
    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))  
data_files = []
fp = open(cf.user_name+cf.file_path_test)
lines = fp.readlines()
fp.close()   
for line in lines:
    line = line.strip('\n')
    data_files.append(line)
c=0
for x,y in ds_test:
    generate_images(generator_g,x,cf,epoch,'pred',data_files[c])
    c+=1
generator_g.save(cf.save_path)