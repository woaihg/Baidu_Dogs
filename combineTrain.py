import h5py
import numpy as np
import keras
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from keras.utils.np_utils import to_categorical
import pandas as pd
from keras.preprocessing.image import *
from keras import backend as K
fileDir = '/home/deeplearning/wh/baiduImage/data/'

def train(trainfiles,testfiles,epoch,train_generator,test_generator,redir):
    
    X_train = []
    X_test = []
    for filename in testfiles:
        with h5py.File(filename, 'r') as h:
            X_test.append(np.array(h['test']))
    
    for filename in trainfiles:
        with h5py.File(filename, 'r') as h:
            X_train.append(np.array(h['test']))

		
    y_train = to_categorical(y_train, num_classes=97)
    X_train = np.concatenate(X_train, axis=1)
    X_train, y_train = shuffle(X_train, y_train)
    X_test = np.concatenate(X_test, axis=1)

    input_tensor = Input(X_train.shape[1:])
    x = Dense(512,activation='relu')(input_tensor)
    x = Dropout(0.5)(x)
    x = Dense(97,activation='softmax')(x)
    model4 = Model(input_tensor,x)

    model4.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['accuracy'])
    model4.fit(X_train, y_train, batch_size=64,nb_epoch=epoch,validation_split=0.0,verbose=True)


    y_pred = model4.predict(X_test, verbose=1)
    y_class=[]
    for i in range(29282):
        y_class.append(y_pred[i].argmax())

    a=train_generator.class_indices
    map={}
    for i in a:
        map[a[i]]=i


    re=[]
    for i, fname in enumerate(test_generator.filenames):
        fname = fname.split('.')[0].split('/')[1]
        re.append([fname,map[y_class[i]]])

    df=pd.DataFrame(re)
    df.to_csv(redir, header=None,index=None)

def main():
    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory(fileDir+"test", (512,512), shuffle=False, batch_size=8, class_mode=None)
    train_generator = gen.flow_from_directory(fileDir+"train_hun", (512,512), shuffle=False, batch_size=8)
    train(['feature_refeature_re/train_nocut_hun/224_1p_densenet161'],['feature_re/test_nocut/224_densenet161'],epoch=8,train_generator,test_generator,'feature_re/re_hun/nocut1p_161.csv')
    train(['feature_re/train_nocut_hun/224_1p_densenet169'],['feature_re/test_nocut/224_densenet169'],epoch=9,train_generator,test_generator,'feature_re/re_hun/nocut1p_169.csv')
    train(['feature_re/train_nocut_hun/224_1p_densenet201'],['feature_re/test_nocut/224_densenet201'],epoch=8,train_generator,test_generator,'feature_re/re_hun/nocut1p_201.csv')
    train(['feature_re/train_nocut_hun/224_1p_resnet101'],['feature_re/test_nocut/224_resnet101'],epoch=14,train_generator,test_generator,'feature_re/re_hun/nocut1p_res101.csv')
    train(['feature_re/train_nocut_hun/448_1p-res152.h5'],['feature_re/test_nocut/448-res152.h5'],epoch=14,train_generator,test_generator,'feature_re/re_hun/nocut1p_res152.csv')
    train(['feature_re/train_nocut_hun/299_1p-incepv4.h5'],['feature_re/test_nocut/299-incepv4.h5'],epoch=14,train_generator,test_generator,'feature_re/re_hun/nocut1p_incepv4.csv')
    train(['feature_re/train_nocut_hun/512_1p-xcep.h5'],['feature_re/test_nocut/512-xcep.h5'],epoch=14,train_generator,test_generator,'re_hun/nocut1p_xcep.csv')
    train(['feature_re/train_nocut_hun/224_1p_densenet161','feature_re/train_nocut_hun/224_1p_resnet101','feature_re/train_nocut_hun/448_1p-res152.h5','feature_re/train_nocut_hun/299_1p-incepv4.h5','feature_re/train_nocut_hun/512_1p-xcep.h5'],
          ['feature_re/test_nocut/224_densenet161','feature_re/test_nocut/224_resnet101','feature_re/test_nocut/448-res152.h5','feature_re/test_nocut/299-incepv4.h5','feature_re/test_nocut/512-xcep.h5'],epoch=6,train_generator,test_generator,'feature_re/re_hun/nocut1p_all8.csv')
    
    train_generator = gen.flow_from_directory(fileDir+"train2p_hun", (512,512), shuffle=False, batch_size=8)
    train(['feature_re/train_nocut_hun/224_2p_densenet161'],['feature_re/test_nocut/224_densenet161'],epoch=8,train_generator,test_generator,'feature_re/re_hun/nocut2p_161.csv')
    train(['feature_re/train_nocut_hun/224_2p_densenet169'],['feature_re/test_nocut/224_densenet169'],epoch=8,train_generator,test_generator,'feature_re/re_hun/nocut2p_169.csv')
    train(['feature_re/train_nocut_hun/224_2p_densenet201'],['feature_re/test_nocut/224_densenet201'],epoch=8,train_generator,test_generator,'feature_re/re_hun/nocut2p_201.csv')
    train(['feature_re/train_nocut_hun/224_2p_resnet101'],['feature_re/test_nocut/224_resnet101'],epoch=11,train_generator,test_generator,'feature_re/re_hun/nocut2p_res101.csv')
    train(['feature_re/train_nocut_hun/448_2p-res152.h5'],['feature_re/test_nocut/448-res152.h5'],epoch=11,train_generator,test_generator,'feature_re/re_hun/nocut2p_res152.csv')
    train(['feature_re/train_nocut_hun/299_2p-incepv4.h5'],['feature_re/test_nocut/299-incepv4.h5'],epoch=11,train_generator,test_generator,'feature_re/re_hun/nocut2p_incepv4.csv')
    train(['feature_re/train_nocut_hun/512_2p-xcep.h5'],['feature_re/test_nocut/512-xcep.h5'],epoch=11,train_generator,test_generator,'feature_re/re_hun/nocut2p_xcep.csv')
    train(['feature_re/train_nocut_hun/224_2p_densenet161','feature_re/train_nocut_hun/224_2p_resnet101','feature_re/train_nocut_hun/448_2p-res152.h5','feature_re/train_nocut_hun/299_2p-incepv4.h5','feature_re/train_nocut_hun/512_2p-xcep.h5'],
          ['feature_re/test_nocut/224_densenet161','feature_re/test_nocut/224_resnet101','feature_re/test_nocut/448-res152.h5','feature_re/test_nocut/299-incepv4.h5','feature_re/test_nocut/512-xcep.h5'],epoch=6,train_generator,test_generator,'feature_re/re_hun/nocut2p_all8.csv')
    
    train_generator = gen.flow_from_directory(fileDir+"train3p_hun", (512,512), shuffle=False, batch_size=8)
    train(['feature_re/train_nocut_hun/448_3p-res152.h5'],['feature_re/test_nocut/448-res152.h5'],epoch=10,train_generator,test_generator,'feature_re/re_hun/nocut3p_res152.csv')
    train(['feature_re/train_nocut_hun/299_3p-incepv4.h5'],['feature_re/test_nocut/299-incepv4.h5'],epoch=10,train_generator,test_generator,'feature_re/re_hun/nocut3p_incepv4.csv')
    train(['feature_re/train_nocut_hun/512_3p-xcep.h5'],['feature_re/test_nocut/512-xcep.h5'],epoch=10,train_generator,test_generator,'feature_re/re_hun/nocut3p_xcep.csv')
    train(['feature_re/train_nocut_hun/448_3p-res152.h5','feature_re/train_nocut_hun/299_3p-incepv4.h5','feature_re/train_nocut_hun/512_3p-xcep.h5'],
          ['feature_re/test_nocut/448-res152.h5','feature_re/test_nocut/299-incepv4.h5','feature_re/test_nocut/512-xcep.h5'],epoch=5,train_generator,test_generator,'feature_re/re_hun/nocut3p_all4.csv')
    
    test_generator = gen.flow_from_directory(fileDir+"test_head", (512,512), shuffle=False, batch_size=8, class_mode=None)
    train_generator = gen.flow_from_directory(fileDir+"train_hun_head", (512,512), shuffle=False, batch_size=8)
    train(['feature_re/train_head_hun/224_densenet161'],['feature_re/test_head/224_densenet161'],epoch=20,train_generator,test_generator,'feature_re/re_hun/head_161.csv')
    train(['feature_re/train_head_hun/224_resnet101'],['feature_re/test_head/224_resnet101'],epoch=24,train_generator,test_generator,'feature_re/re_hun/head_res101.csv')
    train(['feature_re/train_head_hun/512-xcep.h5'],['feature_re/test_head/512-xcep.h5'],epoch=24,train_generator,test_generator,'feature_re/re_hun/head_xcep.csv')
    
    test_generator = gen.flow_from_directory(fileDir+"test_cut", (512,512), shuffle=False, batch_size=8, class_mode=None)
    train_generator = gen.flow_from_directory(fileDir+"train_cut_hun", (512,512), shuffle=False, batch_size=8)
    train(['feature_re/train_cut_hun/224_1p_densenet161'],['feature_re/test_cut/224_densenet161'],epoch=8,train_generator,test_generator,'feature_re/re_hun/cut1p_161.csv')
    train(['train_cut_hun/224_1p_densenet169'],['test_cut/224_densenet169'],epoch=9,train_generator,test_generator,'feature_re/re_hun/cut1p_169.csv')
    train(['train_cut_hun/224_1p_densenet201'],['test_cut/224_densenet201'],epoch=8,train_generator,test_generator,'feature_re/re_hun/cut1p_201.csv')
    train(['train_cut_hun/224_1p_resnet101'],['test_cut/224_resnet101'],epoch=14,train_generator,test_generator,'feature_re/re_hun/cut1p_res101.csv')
    train(['train_cut_hun/448_1p-res152.h5'],['test_cut/448-res152.h5'],epoch=14,train_generator,test_generator,'feature_re/re_hun/cut1p_res152.csv')
    train(['train_cut_hun/299_1p-incepv4.h5'],['test_cut/299-incepv4.h5'],epoch=14,train_generator,test_generator,'feature_re/re_hun/cut1p_incepv4.csv')
    train(['train_cut_hun/512_1p-xcep.h5'],['test_cut/512-xcep.h5'],epoch=14,train_generator,test_generator,'feature_re/re_hun/cut1p_xcep.csv')
    train(['feature_re/train_cut_hun/224_1p_densenet161','feature_re/train_cut_hun/224_1p_resnet101','feature_re/train_cut_hun/448_1p-res152.h5','feature_re/train_cut_hun/299_1p-incepv4.h5','feature_re/train_cut_hun/512_1p-xcep.h5'],
          ['feature_re/test_cut/224_densenet161','feature_re/test_cut/224_resnet101','feature_re/test_cut/448-res152.h5','feature_re/test_cut/299-incepv4.h5','feature_re/test_cut/512-xcep.h5'],epoch=6,train_generator,test_generator,'feature_re/re_hun/cut1p_all8.csv')
    
    train_generator = gen.flow_from_directory(fileDir+"train_cut2p_hun", (512,512), shuffle=False, batch_size=8)
    train(['feature_re/train_cut_hun/448_2p-res152.h5'],['feature_re/test_cut/448-res152.h5'],epoch=11,train_generator,test_generator,'feature_re/re_hun/cut2p_res152.csv')
    train(['feature_re/train_cut_hun/299_2p-incepv4.h5'],['feature_re/test_cut/299-incepv4.h5'],epoch=11,train_generator,test_generator,'feature_re/re_hun/cut2p_incepv4.csv')
    train(['feature_re/train_cut_hun/512_2p-xcep.h5'],['feature_re/test_cut/512-xcep.h5'],epoch=11,train_generator,test_generator,'feature_re/re_hun/cut2p_xcep.csv')
    train(['feature_re/train_cut_hun/448_2p-res152.h5','feature_re/train_cut_hun/299_2p-incepv4.h5','feature_re/train_cut_hun/512_2p-xcep.h5'],
          ['feature_re/test_cut/448-res152.h5','feature_re/test_cut/299-incepv4.h5','feature_re/test_cut/512-xcep.h5'],epoch=6,train_generator,test_generator,'feature_re/re_hun/cut2p_all4.csv')
    
    train_generator = gen.flow_from_directory(fileDir+"train_cut3p_hun", (512,512), shuffle=False, batch_size=8)
    train(['feature_re/train_cut_hun/448_3p-res152.h5'],['feature_re/test_cut/448-res152.h5'],epoch=10,train_generator,test_generator,'feature_re/re_hun/cut3p_res152.csv')
    train(['feature_re/train_cut_hun/299_3p-incepv4.h5'],['feature_re/test_cut/299-incepv4.h5'],epoch=10,train_generator,test_generator,'feature_re/re_hun/cut3p_incepv4.csv')
    train(['feature_re/train_cut_hun/512_3p-xcep.h5'],['feature_re/test_cut/512-xcep.h5'],epoch=10,train_generator,test_generator,'feature_re/re_hun/cut3p_xcep.csv')
    train(['feature_re/train_cut_hun/448_3p-res152.h5','feature_re/train_cut_hun/299_3p-incepv4.h5','feature_re/train_cut_hun/512_3p-xcep.h5'],
          ['feature_re/test_cut/448-res152.h5','feature_re/test_cut/299-incepv4.h5','feature_re/test_cut/512-xcep.h5'],epoch=5,train_generator,test_generator,'feature_re/re_hun/cut3p_all4.csv')
    
if __name__ == '__main__':
    main()


