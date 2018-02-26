from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import h5py
import inceptionV4
import resnet152
import densenet161
import densenet121
import densenet169
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
import numpy as np
import cv2
import os
import math
fileDir = '/home/deeplearning/wh/baiduImage/data/'

def get_processed_image1(img_path,img_h,img_w):
    # Load image and convert from BGR to RGB
    im = np.asarray(cv2.imread(img_path))[:,:,::-1]
    im = cv2.resize(im, (img_h, img_w))
    return im

def get_processed_image2(img_path,img_h,img_w):
    # Load image and convert from BGR to RGB
    im = np.asarray(cv2.imread(img_path))[:,:,::-1]
    im = cv2.resize(im, (img_h, img_w))
    im = inceptionV4.preprocess_input(im)
    return im



def xcep():
    input_tensor = Input((512, 512, 3))
    x = input_tensor
   
    x = Lambda(xception.preprocess_input)(x)
    base_model = Xception(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    batchsize=16
    image_size=(512,512)
    batch_X = np.zeros((batchsize,)+(512,512,3),dtype=K.floatx())
    train_feature=[]
    gen = ImageDataGenerator()
    train = gen.flow_from_directory(fileDir+"train_hun", image_size, shuffle=False, batch_size=8)#18686
    postrain = fileDir+'train_hun_head/'
    for idx in range(0, len(train.filenames),batchsize):
        print(idx)
        if idx + batchsize<len(train.filenames):
            for i in range(batchsize):
                batch_X[i]=get_processed_image1(postrain+train.filenames[idx+i].split('/')[1],512,512)
            train_feature.append(model.predict_on_batch(batch_X))
        else:
            length = len(train.filenames)-idx
            for i in range(length):
                batch_X[i]=get_processed_image1(postrain+train.filenames[idx+i].split('/')[1],512,512)
            train_feature.append(model.predict_on_batch(batch_X[0:length]))

    train_feature = np.array(train_feature)
    train_feature = np.concatenate(train_feature, 0)
    
    test_feature=[]
    test = gen.flow_from_directory(fileDir+"test", image_size, shuffle=False, batch_size=8)#18686
    postest = fileDir+'test_head/'
    for idx in range(0, len(test.filenames),batchsize):
        print(idx)
        if idx + batchsize<len(test.filenames):
            for i in range(batchsize):
                batch_X[i]=get_processed_image1(postest+test.filenames[idx+i].split('/')[1],512,512)
            test_feature.append(model.predict_on_batch(batch_X))
        else:
            length = len(test.filenames)-idx
            for i in range(length):
                batch_X[i]=get_processed_image1(postest+test.filenames[idx+i].split('/')[1],512,512)
            test_feature.append(model.predict_on_batch(batch_X[0:length]))

    test_feature = np.array(test_feature)
    test_feature = np.concatenate(test_feature, 0)

    print(train_feature.shape)
    print(test_feature.shape)

    with h5py.File(fileDir+'train_head_hun/512-xcep.h5', "w") as f:
         f.create_dataset("train", data=train_feature)

    with h5py.File(fileDir+'test_head/512-xcep.h5', "w") as f:
         f.create_dataset("test", data=test_feature)


def main():
    xcep() 

if __name__ == '__main__':
    main()
