'''
An example to show the interface.
'''
from skimage import io
from numpy import linalg as LA
import numpy as np
import scipy.io as sio
import skimage.io
import os
import random
import sys
GPU_id = 0
caffe_root = 'I:/Caffe/Server/caffe-master/' # change this to your own pycaffe path
sys.path.insert(0, caffe_root)
import caffe
caffe.set_mode_gpu()
caffe.set_device(GPU_id)
#from '../model/LightenedCNN_B_caffe.caffemodel' import load_model

# Note to load your model outside of `FaceVerification` function,
# otherwise, model will be loaded every comparison, which is too time-consuming.
#model = load_model()

#get feature
def extract_feat(protopath,modelpath,imgpath,layer):
    # caffe.set_mode_cpu()
    net = caffe.Net(protopath, modelpath, caffe.TEST)
    if layer not in net.blobs:
        raise TypeError("Invalid layer name: " + layer)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.array([0.7]))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 1)

    feats = []
    #i = 0
    img = caffe.io.load_image(imgpath, False)
    img = caffe.io.resize_image(img,(128,128))
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    output=net.forward()
    feat = net.blobs[layer].data[0]
    feats.append(feat/LA.norm(feat))
    return np.array(feats)


# model compare
def cos_distance(feat1,feat2):    
    sum = np.dot(feat1,feat2)
    square1 = np.square(feat1)
    square2 = np.square(feat2)    
    sum1 = square1.sum()
    sum2 = square2.sum()    
    sqrt1 = np.sqrt(sum1)
    sqrt2 = np.sqrt(sum2)       
    cos_dis = sum / (sqrt1 * sqrt2)   
    return cos_dis

def FaceVerification(img_path1, img_path2):
    protopath='LightenedCNN_A_deploy.prototxt'
    modelpath='centerloss_finetune_144000_iter_140000.caffemodel'
    feat_img1 = extract_feat(protopath, modelpath, img_path1, 'fc1')
    feat_img2 = extract_feat(protopath, modelpath, img_path2, 'fc1')
    distance = cos_distance(feat_img1[0], feat_img2[0])
    if distance > 0.45:
        return 1
    else:
        return 0