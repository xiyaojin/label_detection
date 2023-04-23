# -*- coding: utf-8 -*-
from comet_ml import Experiment
import tensorflow as tf
import shutil
import time
from models import AE,Res
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

experiment=Experiment(api_key='YmSX7HeUw7Lcx3KN18iQe47yi',project_name='autoencoder',workspace='xiyaojin')
experiment.set_name(cf.experiment_name)



ds_train=DataGenerator(cf,cf.file_path_train,shuffle=True,aug=False)
#ds_test=DataGenerator(cf,cf.file_path_test,shuffle=False,aug=False)
ds_valid=DataGenerator(cf,cf.file_path_validation,shuffle=True,aug=False)

fp = open(cf.user_name+cf.file_path_train)
lines = fp.readlines()
N=len(lines)
fp.close()
fp = open(cf.user_name+cf.file_path_validation)
lines = fp.readlines()
N_val=len(lines)
fp.close()      

model=Res()

lr=tf.keras.optimizers.schedules.ExponentialDecay(cf.lr,decay_steps=-50*N,decay_rate=np.exp(1))
optimizer = tf.keras.optimizers.SGD(cf.lr, momentum=0.9)

checkpoint_path = cf.save_path
ckpt = tf.train.Checkpoint(model=model,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.

starting_epoch=0
if cf.resume==True:
    starting_epoch=cf.starting_epoch
    status=ckpt.restore(os.path.join(cf.save_path,cf.ckpt_file))
    #status=ckpt.restore(ckpt_manager.latest_checkpoint)
    status.assert_existing_objects_matched()
    print ('Latest checkpoint restored!!')

    
def generate_images(gen,test_input,cf,epoch,phase,name=''):
    pred=gen(test_input)
    pred=np.squeeze(pred)
    pred=np.argmax(pred,axis=-1)
    pred=np.transpose(pred,[2,0,1])
    pred_sitk=sitk.GetImageFromArray(pred)
    pred_sitk=sitk.Cast(pred_sitk,sitk.sitkFloat32)   

    if phase=='pred':
        save_path=os.path.join(cf.save_path,'generated_images')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sitk.WriteImage(pred_sitk,os.path.join(save_path,name+'.nii.gz'))
    else:
        sitk.WriteImage(pred_sitk,os.path.join(cf.save_path,phase+'_'+str(epoch+1)+'.nii.gz'))

@tf.function
def train_step(x, y):
    with tf.GradientTape(persistent=True) as tape:      
        pred = model(x)        
        loss = mdl(y,pred)
        acc = accuracy(y,pred)
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    return loss,acc

@tf.function
def valid_step(x,y):
    pred = model(x)
    acc = accuracy(y,pred)
    return acc

for i in range(EPOCHS):
    epoch=i+starting_epoch
    start = time.time()
    print('epoch:'+str(epoch+1))
    temp_loss=0
    temp_acc=0
    temp_val_acc=0
    n = 0
    for image_x, image_y in ds_train:
        #print (tf.shape(image_x),tf.shape(image_y))
        loss,acc=train_step(image_x, image_y)
        print(str(n+1)+'/'+str(N)+',','loss:',loss.numpy(),'acc:',acc.numpy(),
              end='\r')
        n+=1
        temp_loss+=loss.numpy()
        temp_acc+=acc.numpy()
    experiment.log_metric('loss',temp_loss/N,step=epoch*N+n)
    experiment.log_metric('acc',temp_acc/N,step=epoch*N+n)
    experiment.log_metric('epoch',epoch+1,step=epoch*N+n)
    print('\n')
    for x,y in ds_valid:
        val_acc=valid_step(x,y)
        print('val_acc:',val_acc.numpy(),
              end='\r')
        temp_val_acc+=val_acc
    experiment.log_metric('val_acc',temp_val_acc/N_val,step=epoch*N+n)        

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
        generate_images(model,train_sample,cf,epoch,'train')
        test_sample=next(iter(ds_valid))[0]
        generate_images(model,test_sample,cf,epoch,'valid')        
        #generate_images_from_patch(generator_g,ds_test,cf,epoch)
    
    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))  
    
model.save(cf.save_path)
# data_files = []
# fp = open(cf.user_name+cf.file_path_test)
# lines = fp.readlines()
# fp.close()   
# for line in lines:
#     line = line.strip('\n')
#     data_files.append(line)
# c=0
# for x,y in ds_test:
#     generate_images(generator_g,x,cf,epoch,'pred',data_files[c])
#     c+=1
# generator_g.save(cf.save_path)