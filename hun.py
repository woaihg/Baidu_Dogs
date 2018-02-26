import numpy as np
import os
import shutil
import h5py
import random
import pandas as pd
Dict={}

with open('1598.txt','r') as f:
    data = f.readlines()
    for line in data:
        b = line.split('\t')
        print(b[1])
        if b[0]=='46':
            b[0]='31'
        if b[0]=='111':
            b[0]='87'
        if b[0]=='74':
            b[0]='72'
        if os.path.exists('data/train_hun/'+b[0]+'/'+b[1])==False:
            shutil.copy("data/train/"+b[1],'data/train_hun/'+b[0]+'/'+b[1])


