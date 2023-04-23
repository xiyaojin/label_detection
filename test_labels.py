# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import SimpleITK as sitk
from argparse import ArgumentParser
import imp
from process_config import process_config
import os

def dice(pred,seg):
    dsc=0
    smooth=0.0000001
    for i in range(9):
        a=(pred==(i+1)).astype(np.int)
        b=(seg==(i+1)).astype(np.int)
        dsc+=2*(np.sum(a*b)+smooth)/(np.sum(a)+np.sum(b)+smooth)
    return dsc/9

parser=ArgumentParser(description='Model testing')

parser.add_argument('-c', '--config_file', help='experiment config file')
args = parser.parse_args()
assert args.config_file is not None, 'Please provide an experiment configuration'\
                                    'file using -c /path/to/config.py'\
                                    ' in the command line'

config_path=args.config_file
cf=imp.load_source('config',config_path)
cf=process_config(cf)
print('start loading model..')
model=tf.keras.models.load_model(cf.save_path)
print('model loaded!')

img=sitk.ReadImage(r'/home/xjin/ticker3D/ticker/nii_patient68.nii')
img=np.transpose(sitk.GetArrayFromImage(img),axes=[1,2,0])


#img=np.load(r'/home/xjin/data/data.3D_2mm_fixed/labels/patient1.npy')
#img=np.round(img/255.0*9)
img=np.expand_dims(img, axis=(0,-1))
pred=model.predict(img)
pred=np.squeeze(pred)
pred=np.argmax(pred,axis=-1)

dsc=dice(pred,np.squeeze(img))
print ('dsc=',dsc)
pred=np.transpose(pred,axes=[2,0,1])
pred=sitk.GetImageFromArray(pred)
pred=sitk.Cast(pred,sitk.sitkFloat32)
sitk.WriteImage(pred,os.path.join(cf.save_path,'AE-68.nii.gz'))
