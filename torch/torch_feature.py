from __future__ import print_function

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch

import torchvision
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os, sys, h5py, gc, argparse, codecs, shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
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
torch.cuda.set_device(1)
parser = argparse.ArgumentParser()
parser.add_argument('--ffpath', required=True, help='path for feature file')
parser.add_argument('--model', required=True, help='cnn model')
parser.add_argument('--crop', required=False, action='store_true', help='dog detection')
opt = parser.parse_args()
print(opt)

print('train')

# dog_crop = h5py.File('./yolo_kuhuang.h5', 'r')
# dog_crop_img = dog_crop.keys()
def readtrain_img(img_file, size = (224, 224), logging = False, postrain):
    imgs = []
    for img_path in img_file:
        #a = img_path.split('/')[1]
        img = Image.open(postrain+img_path)
        imgs.append(img)
    return imgs

def readtest_img(img_file, size = (224, 224), logging = False,postest):
    imgs = []
    for img_path in img_file:
        #a = img_path.split('/')[1]
        img = Image.open(postest+img_path)
        imgs.append(img)
    return imgs

network = opt.model
print('model',network)
if network == 'resnet18':
    model_conv = torchvision.models.resnet18(pretrained=True)
    model_conv = nn.Sequential(*list(model_conv.children())[:-1])
    featurenum = 512
    batchsize = 80
elif network == 'resnet34':
    model_conv = torchvision.models.resnet34(pretrained=True)
    model_conv = nn.Sequential(*list(model_conv.children())[:-1])
    featurenum = 512
    batchsize = 40
elif network == 'resnet50':
    model_conv = torchvision.models.resnet50(pretrained=True)
    model_conv = nn.Sequential(*list(model_conv.children())[:-1])
    featurenum = 2048
    batchsize = 25
elif network == 'resnet101':
    model_conv = torchvision.models.resnet101(pretrained=True)
    model_conv = nn.Sequential(*list(model_conv.children())[:-1])
    featurenum = 2048
    batchsize = 20
elif network == 'resnet152':
    model_conv = torchvision.models.resnet152(pretrained=True)
    model_conv = nn.Sequential(*list(model_conv.children())[:-1])
    featurenum = 2048
    batchsize = 10
elif network == 'vgg11':
    model_conv = torchvision.models.vgg11(pretrained=True)
    model_conv.classifier = nn.Sequential(*list(model_conv.classifier.children())[:-1])
    featurenum = 4096
    batchsize = 34
elif network == 'vgg13':
    model_conv = torchvision.models.vgg13(pretrained=True)
    model_conv.classifier = nn.Sequential(*list(model_conv.classifier.children())[:-1])
    featurenum = 4096
    batchsize = 34
elif network == 'vgg16':
    model_conv = torchvision.models.vgg16(pretrained=True)
    model_conv.classifier = nn.Sequential(*list(model_conv.classifier.children())[:-1])
    featurenum = 4096
    batchsize = 34
elif network == 'vgg19':
    model_conv = torchvision.models.vgg19(pretrained=True)
    model_conv.classifier = nn.Sequential(*list(model_conv.classifier.children())[:-1])
    featurenum = 4096
    batchsize = 30
elif network == 'densenet121':
    model_conv = torchvision.models.densenet121(pretrained=True)
    print('model load')
    model_conv.classifier = nn.Sequential(*list(model_conv.classifier.children())[:-1])
    featurenum = 1024
    batchsize = 25
elif network == 'densenet161':   
    model_conv = torchvision.models.densenet161(pretrained=True)
    print('model load')
    model_conv.classifier = nn.Sequential(*list(model_conv.classifier.children())[:-1])
    featurenum = 2208
    batchsize = 10
elif network == 'densenet169':
    model_conv = torchvision.models.densenet169(pretrained=True)
    print('model load')
    model_conv.classifier = nn.Sequential(*list(model_conv.classifier.children())[:-1])
    featurenum = 1664
    batchsize = 10
elif network == 'densenet201':
    model_conv = torchvision.models.densenet201(pretrained=True)
    model_conv.classifier = nn.Sequential(*list(model_conv.classifier.children())[:-1])
    featurenum = 1920
    batchsize = 15
elif network == 'inception':
    model_conv = torchvision.models.inception_v3(pretrained = True, transform_input=False)
    # model_conv.classifier = nn.Sequential(*list(model_conv.classifier.children())[:-1])
    featurenum = 1000
    batchsize = 35


model_conv.eval().cuda(1)
print(network, featurenum)


if network == 'inception':
    tr = transforms.Compose([
            transforms.Scale(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                 std = [ 0.229, 0.224, 0.225 ])
    ])
else:
    tr = transforms.Compose([
            transforms.Scale(224),
            # transforms.RandomSizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                 std = [ 0.229, 0.224, 0.225 ])
    ])

gen = ImageDataGenerator()
train = gen.flow_from_directory("./data/train_hun", (224,224), shuffle=False, batch_size=batchsize, class_mode=None)
train_feature = []

for idx in range(0, len(train.filenames),batchsize):
    print(idx)
    if idx + batchsize<len(train.filenames):
        ff = readtrain_img(train.filenames[idx:idx+batchsize],postrain='./data/train_hun/')
        ff = [tr(x) for x in ff]
        ff = torch.stack(ff)
        ff = model_conv(Variable(ff.cuda(1))).view(-1,featurenum)        
        train_feature.append(ff.data.cpu().numpy())
        del ff;gc.collect()
    else:
        ff = readtrain_img(train.filenames[idx:],postrain='./data/train_hun/')
        ff = [tr(x) for x in ff]
        ff = torch.stack(ff)
        ff = model_conv(Variable(ff.cuda(1))).view(-1,featurenum)
        train_feature.append(ff.data.cpu().numpy())
        del ff;gc.collect()
train_feature = np.array(train_feature)

test = gen.flow_from_directory("./data/test", (224,224), shuffle=False, batch_size=8, class_mode=None)

test_feature = []
for idx in range(0, len(test.filenames),batchsize):
    print(idx)
    if idx + batchsize<len(test.filenames):
        
        ff = readtest_img(test.filenames[idx:idx+batchsize],postest='./data/test/')
        ff = [tr(x) for x in ff]
        ff = torch.stack(ff)
        ff = model_conv(Variable(ff.cuda(1))).view(-1,featurenum)
        test_feature.append(ff.data.cpu().numpy())
        del ff;gc.collect()
    else:
        ff = readtest_img(test.filenames[idx:],postest='./data/test/')
        ff = [tr(x) for x in ff]
        ff = torch.stack(ff)
        ff = model_conv(Variable(ff.cuda(1))).view(-1,featurenum)
        test_feature.append(ff.data.cpu().numpy())
        del ff;gc.collect()
test_feature = np.array(test_feature)

train_feature = np.concatenate(train_feature, 0).reshape(-1, featurenum)
test_feature = np.concatenate(test_feature, 0).reshape(-1, featurenum)

with h5py.File('./feature_re/train_nocut_hun/224_1p_'+network, "w") as f:
    f.create_dataset("train", data=train_feature)

with h5py.File('./feature_re/test_nocut/224_'+network, "w") as f:
    f.create_dataset("train", data=train_feature)
   
    
    
    
train = gen.flow_from_directory("./data/train2p_hun", (224,224), shuffle=False, batch_size=batchsize, class_mode=None)
train_feature = []
for idx in range(0, len(train.filenames),batchsize):
    print(idx)
    if idx + batchsize<len(train.filenames):
        ff = readtrain_img(train.filenames[idx:idx+batchsize],postrain='./data/train2p_hun/')
        ff = [tr(x) for x in ff]
        ff = torch.stack(ff)
        ff = model_conv(Variable(ff.cuda(1))).view(-1,featurenum)        
        train_feature.append(ff.data.cpu().numpy())
        del ff;gc.collect()
    else:
        ff = readtrain_img(train.filenames[idx:],postrain='./data/train2p_hun/')
        ff = [tr(x) for x in ff]
        ff = torch.stack(ff)
        ff = model_conv(Variable(ff.cuda(1))).view(-1,featurenum)
        train_feature.append(ff.data.cpu().numpy())
        del ff;gc.collect()
train_feature = np.array(train_feature)
train_feature = np.concatenate(train_feature, 0).reshape(-1, featurenum)
with h5py.File('./feature_re/train_nocut_hun/224_2p_'+network, "w") as f:
    f.create_dataset("train", data=train_feature)
    
    

train = gen.flow_from_directory("./data/train_cut_hun", (224,224), shuffle=False, batch_size=batchsize, class_mode=None)
train_feature = []

for idx in range(0, len(train.filenames),batchsize):
    print(idx)
    if idx + batchsize<len(train.filenames):
        ff = readtrain_img(train.filenames[idx:idx+batchsize],postrain='./data/train_cut_hun/')
        ff = [tr(x) for x in ff]
        ff = torch.stack(ff)
        ff = model_conv(Variable(ff.cuda(1))).view(-1,featurenum)        
        train_feature.append(ff.data.cpu().numpy())
        del ff;gc.collect()
    else:
        ff = readtrain_img(train.filenames[idx:],postrain='./data/train_cut_hun/')
        ff = [tr(x) for x in ff]
        ff = torch.stack(ff)
        ff = model_conv(Variable(ff.cuda(1))).view(-1,featurenum)
        train_feature.append(ff.data.cpu().numpy())
        del ff;gc.collect()
train_feature = np.array(train_feature)

test = gen.flow_from_directory("./data/test_cut", (224,224), shuffle=False, batch_size=8, class_mode=None)

test_feature = []
for idx in range(0, len(test.filenames),batchsize):
    print(idx)
    if idx + batchsize<len(test.filenames):
        
        ff = readtest_img(test.filenames[idx:idx+batchsize],postest='./data/test_cut/')
        ff = [tr(x) for x in ff]
        ff = torch.stack(ff)
        ff = model_conv(Variable(ff.cuda(1))).view(-1,featurenum)
        test_feature.append(ff.data.cpu().numpy())
        del ff;gc.collect()
    else:
        ff = readtest_img(test.filenames[idx:],postest='./data/test_cut/')
        ff = [tr(x) for x in ff]
        ff = torch.stack(ff)
        ff = model_conv(Variable(ff.cuda(1))).view(-1,featurenum)
        test_feature.append(ff.data.cpu().numpy())
        del ff;gc.collect()
test_feature = np.array(test_feature)

train_feature = np.concatenate(train_feature, 0).reshape(-1, featurenum)
test_feature = np.concatenate(test_feature, 0).reshape(-1, featurenum)

with h5py.File('./feature_re/train_cut_hun/224_1p_'+network, "w") as f:
    f.create_dataset("train", data=train_feature)

with h5py.File('./feature_re/test_cut/224_'+network, "w") as f:
    f.create_dataset("train", data=train_feature)