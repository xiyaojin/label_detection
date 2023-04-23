# -*- coding: utf-8 -*-
import numpy as np
import os
path=r'C:\Users\xiyaojin\Documents\GitHub\label_detection'
name='test'

f=open(os.path.join(path,name+'.txt'),'w')
for i in range(20):
    f.write('patient'+str(i+101)+'\n')
f.close()