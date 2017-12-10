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

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

n_classes = 2
train = pd.read_csv("/home/kshitij/Desktop/Dataset/stage1_sample_submission.csv")
test = pd.read_csv("/home/kshitij/Desktop/Dataset/stage2_sample_submission.csv")

df = pd.read_csv('/home/kshitij/Desktop/Dataset/stage1_labels.csv')
label_map = df['cancer'].tolist()

base_model = ResNet50(weights='imagenet', pooling=max, include_top = False)
input = Input(shape=(224,224,3),name = 'image_input')
x = base_model(input)
x = Flatten()(x)
model = Model(inputs=input, outputs=x)

X_train = []
y_train = []

X_test = []



# def load_extractor():
#     network = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=None)
#     return network


def load_3d_data(path):
    # credit: https://www.kaggle.com/mumech/loading-and-processing-the-sample-images
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])


def convert_rgb(path):
    CTs = load_3d_data(path)
    CTs[CTs == -2000] = 0  # -2000 is a flag for missing value
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
    #net = load_extractor()
    i = 1
    for folder in glob.glob(path + '/*'):
        if not (os.path.isfile(folder)):
            #print folder
            if not (os.path.isfile(folder + '.npy')):

                img = convert_rgb(folder)
                x = img
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                x = x[0]
                features = model.predict(x)
                features_reduce = features.squeeze()
                my_new_list = [k * 10000 for k in features_reduce]
                features_reduce = my_new_list
                X_test.append(features_reduce)
                np.save(folder, features_reduce)
                print 'Completed:', i
            i += 1


def extract_features_train(path):
    #net = load_extractor()
    i = 1
    for folder in glob.glob(path + '/*'):
        if not (os.path.isfile(folder)):
            if not (os.path.isfile(folder + '.npy')):
                img = convert_rgb(folder)
                x = img
                x = x[0]
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features = model.predict(x)
                features_reduce =  features.squeeze()
                my_new_list = [k * 10000 for k in features_reduce]
                features_reduce = my_new_list
                X_train.append(features_reduce)
                np.save(folder, features_reduce)
                print 'Completed:', i
            i += 1

    X = np.array(X_train)

if __name__ == '__main__':
    extract_features('/home/kshitij/Desktop/Dataset/stage2')
    extract_features_train('/home/kshitij/Desktop/Dataset/stage1')
