# -*- coding: utf-8 -*-
import os
import shutil
def process_config(cf):
    if not hasattr(cf,'user_name'):
        cf.user_name='/home/xjin'
    OUTPUT=cf.user_name+'/label_detection/experiment_output'
    cf.save_path=os.path.join(OUTPUT,cf.experiment_name)
    if not os.path.exists(cf.save_path):
        os.makedirs(cf.save_path)

    shutil.copyfile(cf.user_name+'/label_detection/code/models.py', os.path.join(cf.save_path, "models.py"))
    shutil.copyfile(cf.user_name+'/label_detection/code/layers.py', os.path.join(cf.save_path, "layers.py"))
    shutil.copyfile(cf.user_name+'/label_detection/code/loss_functions.py', os.path.join(cf.save_path, "loss_functions.py"))
    shutil.copyfile(cf.user_name+'/label_detection/code/data_generator.py', os.path.join(cf.save_path, "data_generator.py"))    
    shutil.copyfile(cf.user_name+'/label_detection/code/main.py', os.path.join(cf.save_path, "main.py"))
    if not hasattr(cf,'shape'):
        cf.shape=(100,100,100)

    return cf