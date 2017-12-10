import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
from tqdm import tqdm
import xgboost as xgb
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Flatten, Input
import scipy
from sklearn.metrics import fbeta_score
import dicom
import cv2
import glob
from skimage.feature import hog
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

def load_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def convert_rgb(path):
    CTs = load_3d_data(path)
    CTs[CTs == -2000] = 0
    RGBs = []
    for i in range(0, CTs.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = CTs[i + j]
            img = 255.0 / np.amax(img) * img
            img = np.array(cv2.resize(img, (224, 224)))
            tmp.append(img)
        tmp = np.array(tmp).reshape(224, 224, 3)
        RGBs.append(tmp)
    return np.array(RGBs)


def extract_features(path):
    i = 1
    for folder in glob.glob(path + '/*'):
        if not (os.path.isfile(folder)):
            if not (os.path.isfile(folder + '.npy')):

                img = convert_rgb(folder)
                img = rgb2gray(img)
                maxt =  img.shape[0]
                mint = 0
                val = (maxt-mint)/16
                print maxt
                print mint
                j = 0
                pi = []
                for k in xrange(mint,maxt,val):
                    pi.append(img[k])
                    j+=1
                pi = np.array(pi)
                pi = pi.reshape(224,224*j)
                kp =  hog(pi)
                np.save(folder, kp)
                print 'Completed:', i
            i += 1


if __name__ == '__main__':
    extract_features('/home/kshitij/Desktop/Dataset/stage2')
    extract_features('/home/kshitij/Desktop/Dataset/stage1')
