import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tqdm import tqdm

import numpy as np
import pandas as pd
from keras.models import load_model,Model
from keras import Sequential
from keras.callbacks import EarlyStopping

from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout
from keras.layers import Lambda, Input, GlobalAveragePooling2D, BatchNormalization
from keras.utils import to_categorical
from tensorflow.keras.models import Model
from keras.preprocessing.image import load_img


labels = pd.read_csv('labels.csv')
classes = sorted(list(set(labels['breed'])))
model = load_model('model.h5')
img_size = (331,331,3)

def get_features(model_name, model_preprocessor, input_size, data):

    input_layer = Input(input_size)
    preprocessor = Lambda(model_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs = avg)
    
    feature_maps = feature_extractor.predict(data, verbose=1)
    return feature_maps


from keras.applications.inception_v3 import InceptionV3, preprocess_input
inception_preprocessor = preprocess_input
from keras.applications.xception import Xception, preprocess_input
xception_preprocessor = preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
inc_resnet_preprocessor = preprocess_input
from keras.applications.nasnet import NASNetLarge, preprocess_input
nasnet_preprocessor = preprocess_input

def extract_features(data):
    inception_features = get_features(InceptionV3, inception_preprocessor, img_size, data)
    xception_features = get_features(Xception, xception_preprocessor, img_size, data)
    nasnet_features = get_features(NASNetLarge, nasnet_preprocessor, img_size, data)
    inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocessor, img_size, data)

    final_features = np.concatenate([inception_features,
                                     xception_features,
                                     nasnet_features,
                                     inc_resnet_features],axis=-1)
    
    
    del inception_features
    del xception_features
    del nasnet_features
    del inc_resnet_features
    gc.collect()
    
    
    return final_features

