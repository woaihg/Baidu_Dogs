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

def xcep(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)

    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory(fileDir+"train_hun", image_size, shuffle=False, batch_size=8)#29255
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#29255
    with h5py.File(fileDir+"train_nocut/512_1p-xcep.h5") as h:
        h.create_dataset("train", data=train)
        
    train_generator = gen.flow_from_directory(fileDir+"train2p_hun", image_size, shuffle=False, batch_size=8)#57983
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#57983
    with h5py.File(fileDir+"train_nocut/512_2p-xcep.h5") as h:
        h.create_dataset("train", data=train)
        
    train_generator = gen.flow_from_directory(fileDir+"train3p_hun", image_size, shuffle=False, batch_size=8)#72897
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#72897
    with h5py.File(fileDir+"train_nocut/512_3p-xcep.h5") as h:
        h.create_dataset("train", data=train)
        
    train_generator = gen.flow_from_directory(fileDir+"train_cut_hun", image_size, shuffle=False, batch_size=8)#29255
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#29255
    with h5py.File(fileDir+"train_cut/512_1p-xcep.h5") as h:
        h.create_dataset("train", data=train)
        
    train_generator = gen.flow_from_directory(fileDir+"train_cut2p_hun", image_size, shuffle=False, batch_size=8)#58011
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#58011
    with h5py.File(fileDir+"train_cut/512_2p-xcep.h5") as h:
        h.create_dataset("train", data=train)
        
    train_generator = gen.flow_from_directory(fileDir+"train_cut3p_hun", image_size, shuffle=False, batch_size=8)#85675
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#85675
    with h5py.File(fileDir+"train_cut/512_3p-xcep.h5") as h:
        h.create_dataset("train", data=train)
    
    test_generator = gen.flow_from_directory(fileDir+"test", image_size, shuffle=False, batch_size=8)
    test = model.predict_generator(test_generator,math.ceil(len(test_generator.filenames)/8),verbose=True)
    with h5py.File(fileDir+"test_nocut/512-xcep.h5") as h:
        h.create_dataset("test", data=train)
        
    test_generator = gen.flow_from_directory(fileDir+"test_cut", image_size, shuffle=False, batch_size=8)
    test = model.predict_generator(test_generator,math.ceil(len(test_generator.filenames)/8),verbose=True)
    with h5py.File(fileDir+"test_cut/512-xcep.h5") as h:
        h.create_dataset("test", data=train)
    
    K.clear_session()


def incepV4():
    
    image_size=(299,299)
    base_model = inceptionV4.create_model(weights='imagenet', include_top=False)
    x = AveragePooling2D((8,8), padding='valid')(base_model.output)
    x = Flatten()(x) 
    model = Model(base_model.input,x)
    gen = ImageDataGenerator(preprocessing_function=inceptionV4.preprocess_input)

    train_generator = gen.flow_from_directory(fileDir+"train_hun", image_size, shuffle=False, batch_size=8)#29255
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#29255
    with h5py.File(fileDir+"train_nocut/299_1p-incepv4.h5") as h:
        h.create_dataset("train", data=train)
        
    train_generator = gen.flow_from_directory("train2p_hun", image_size, shuffle=False, batch_size=8)#57983
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#57983
    with h5py.File(fileDir+"train_nocut/299_2p-incepv4.h5") as h:
        h.create_dataset("train", data=train)
        
    train_generator = gen.flow_from_directory(fileDir+"train3p_hun", image_size, shuffle=False, batch_size=8)#72897
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#72897
    with h5py.File(fileDir+"train_nocut/299_3p-incepv4.h5") as h:
        h.create_dataset("train", data=train)
        
    train_generator = gen.flow_from_directory(fileDir+"train_cut_hun", image_size, shuffle=False, batch_size=8)#29255
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#29255
    with h5py.File(fileDir+"train_cut/299_1p-incepv4.h5") as h:
        h.create_dataset("train", data=train)
        
    train_generator = gen.flow_from_directory(fileDir+"train_cut2p_hun", image_size, shuffle=False, batch_size=8)#58011
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#58011
    with h5py.File(fileDir+"train_cut/299_2p-incepv4.h5") as h:
        h.create_dataset("train", data=train)
        
    train_generator = gen.flow_from_directory(fileDir+"train_cut3p_hun", image_size, shuffle=False, batch_size=8)#85675
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#85675
    with h5py.File(fileDir+"train_nocut/299_3p-incepv4.h5") as h:
        h.create_dataset("train", data=train)
    
    test_generator = gen.flow_from_directory(fileDir+"test", image_size, shuffle=False, batch_size=8)
    test = model.predict_generator(test_generator,math.ceil(len(test_generator.filenames)/8),verbose=True)
    with h5py.File(fileDir+"test_nocut/299-incepv4.h5") as h:
        h.create_dataset("test", data=train)
        
    test_generator = gen.flow_from_directory(fileDir+"test_cut", image_size, shuffle=False, batch_size=8)
    test = model.predict_generator(test_generator,math.ceil(len(test_generator.filenames)/8),verbose=True)
    with h5py.File(fileDir+"test_cut/299-incepv4.h5") as h:
        h.create_dataset("test", data=train)

   
    K.clear_session()



def res152():
    
    image_size=(448,448)
    base_model = resnet152.ResNet152(include_top=False, weights='imagenet')
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    print(model.input)
    gen = ImageDataGenerator()
    
    train_generator = gen.flow_from_directory(fileDir+"train_hun", image_size, shuffle=False, batch_size=8)#29255
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#29255
    with h5py.File(fileDir+"train_nocut/448_1p-res152.h5") as h:
        h.create_dataset("train", data=train)
        
    train_generator = gen.flow_from_directory(fileDir+"train2p_hun", image_size, shuffle=False, batch_size=8)#57983
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#57983
    with h5py.File(fileDir+"train_nocut/448_2p-res152.h5") as h:
        h.create_dataset("train", data=train)
        
    train_generator = gen.flow_from_directory(fileDir+"train3p_hun", image_size, shuffle=False, batch_size=8)#72897
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#72897
    with h5py.File(fileDir+"train_nocut/448_3p-res152.h5") as h:
        h.create_dataset("train", data=train)
        
    train_generator = gen.flow_from_directory(fileDir+"train_cut_hun", image_size, shuffle=False, batch_size=8)#29255
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#29255
    with h5py.File(fileDir+"train_cut/448_1p-res152.h5") as h:
        h.create_dataset("train", data=train)
        
    train_generator = gen.flow_from_directory(fileDir+"train_cut2p_hun", image_size, shuffle=False, batch_size=8)#58011
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#58011
    with h5py.File(fileDir+"train_cut/448_2p-res152.h5") as h:
        h.create_dataset("train", data=train)
        
    train_generator = gen.flow_from_directory(fileDir+"train_cut3p_hun", image_size, shuffle=False, batch_size=8)#85675
    train = model.predict_generator(train_generator,math.ceil(len(train_generator.filenames)/8),verbose=True)#85675
    with h5py.File(fileDir+"train_nocut/448_3p-res152.h5") as h:
        h.create_dataset("train", data=train)
    
    test_generator = gen.flow_from_directory(fileDir+"test", image_size, shuffle=False, batch_size=8)
    test = model.predict_generator(test_generator,math.ceil(len(test_generator.filenames)/8),verbose=True)
    with h5py.File(fileDir+"test_nocut/448-res152.h5") as h:
        h.create_dataset("test", data=train)
        
    test_generator = gen.flow_from_directory(fileDir+"test_cut", image_size, shuffle=False, batch_size=8)
    test = model.predict_generator(test_generator,math.ceil(len(test_generator.filenames)/8),verbose=True)
    with h5py.File(fileDir+"test_cut/448-res152.h5") as h:
        h.create_dataset("test", data=train)
    
    K.clear_session()


def main():
    xcep(Xception, (512,512), xception.preprocess_input)
    incepV4()
    res152()
    
if __name__ == '__main__':
    main()
