import numpy as np
import os
import shutil
import h5py
import random

from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array,load_img
fileDir = '/home/deeplearning/wh/baiduImage/data/'



def picadd(datagen,sdes,tdes,num):
    for dir in os.listdir(fileDir+sdes):
    	for filename in os.listdir(fileDir+sdes+dir):
    	    if os.path.exists(fileDir+tdes+dir)==False:
    	         os.mkdir(fileDir+tdes+dir)
    	    shutil.copy(fileDir+sdes+dir+"/"+filename,fileDir+tdes+dir+"/"+filename)
    	    img = load_img(fileDir+tdes+dir+"/"+filename)
    	    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    	    x = x.reshape((1,) + x.shape)  
    	    i=0
    	    for batch in datagen.flow(x, batch_size=1,save_to_dir=fileDir+tdes+dir+"/", save_prefix='dog', save_format='jpg'):
    	        i += 1
    	        if i > 0:
    	            break  

def main():
    datagen = ImageDataGenerator(rotation_range=40,
                     		 shear_range=0.2,                    		
                             zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
    picadd(datagen,'train_hun','train_2p_hun',0)
    picadd(datagen,'train_hun','train3p_hun',1)
    datagen = ImageDataGenerator(rotation_range=40,
                     		  shear_range=0.2,        
                             width_shift_range=0.2,
                             height_shift_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
    picadd(datagen,'train_cut_hun','train_cut2p_hun',0)
    picadd(datagen,'train_cut_hun','train_cut3p_hun',1)
if __name__ == '__main__':
    main()

