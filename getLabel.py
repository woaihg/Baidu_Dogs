import numpy as np
import os
import shutil
import h5py
import random

Dict={}
yinshe={}
with open('data/训练数据.txt','r') as f:
    data = f.readlines()
    
    for line in data:
        b = line.split(' ')
        Dict[b[0]]=b[1]
 
with open('data/训练数据2.txt','r') as f:
    data = f.readlines()
    
    for line in data:
        b = line.split(' ')
        Dict[b[0]]=b[1]
       
print('finish')
#与图像名字对比,移动到指定目录中
fileLoc = 'E:/wh/百度图像/data/train/train/'
label = []
aa=0
for filename in os.listdir(r"E:/wh/百度图像/data/train/train"):
    b = filename.split('.')[0]
    if yinshe.get(Dict[b])==None:
        os.mkdir('E:/wh/百度图像/train/'+str(aa)+'_'+str(Dict[b])+'/')
        #os.mkdir('E:/wh/百度图像/validation/'+str(aa)+'_'+str(Dict[b])+'/')
        yinshe[Dict[b]]=aa
        aa=aa+1
    shutil.copy('E:/wh/百度图像/data/train/train/'+filename,'E:/wh/百度图像/train/'+str(yinshe[Dict[b]])+'_'+str(Dict[b])+'/')
    '''
    if random.randint(0, 5)==0:
        shutil.copy(fileLoc+filename,'E:/wh/百度图像/train/'+str(yinshe[Dict[b]])+'_'+str(Dict[b])+'/')
    else:
        shutil.copy(fileLoc+filename,'E:/wh/百度图像/validation/'+str(yinshe[Dict[b]])+'_'+str(Dict[b])+'/')
    '''
print("train1 finish")    
    
for filename in os.listdir(r"E:/wh/百度图像/data/train2"):
    b = filename.split('.')[0]
    #not conclude
    if Dict.get(b)!=None:  
        if yinshe.get(Dict[b])==None:
            os.mkdir('E:/wh/百度图像/train/'+str(aa)+'_'+str(Dict[b])+'/')
        #os.mkdir('E:/wh/百度图像/validation/'+str(aa)+'_'+str(Dict[b])+'/')
            yinshe[Dict[b]]=aa
            aa=aa+1
        shutil.copy('E:/wh/百度图像/data/train2/'+filename,'E:/wh/百度图像/train/'+str(yinshe[Dict[b]])+'_'+str(Dict[b])+'/')
#print(Dict['216384,2908428553'])

