import streamlit as st
import numpy as np
import pandas as pd
import cv2
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import h5py
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

feat_model = keras.applications.vgg16.VGG16(weights='imagenet')
feat_model = Model(feat_model.input,feat_model.layers[-2].output)
file_name = os.path.dirname(__file__) +'\\model_final1.h5'
model = h5py.File(file_name, "r")
file_name1 = os.path.dirname(__file__) +'\\tokenizer_v1.pickle'
tokenizer = open(file_name1,'rb')
#tokenizer = pickle.load(open( "\\tokenizer_v1.pickle", "rb" ))

def get_word(pred):
    for e,w in tokenizer.word_index.items():
        if w == pred:
            return e

def caption_path(img_path):

    global feat_model,model,tokenizer

    test_pic = cv2.imread(img_path)
    test_pic = cv2.resize(test_pic,(224,224))
    test_pic = test_pic.reshape((1,224,224,3))
    test_pic = preprocess_input(test_pic)

    test_pic = feat_model(test_pic)
    test_pic = np.reshape(test_pic,(4096,))
    test_pic = test_pic.reshape((1,-1))
    words = 'startseq'
    while 'endseq' not in words:
        test_seq = [words]
        test_seq = tokenizer.texts_to_sequences(test_seq)
        test_seq = np.array(test_seq)
        test_seq = pad_sequences(test_seq,maxlen=10)[0]
        test_seq = test_seq.reshape((1,10))

        pred = model([test_pic,test_seq])

        pred = get_word(np.argmax(pred))
        words = words+' '+pred
    words = words.replace('startseq','')
    words = words.replace('endseq','')
    return words.rstrip().lstrip()


def caption_img(test_pic):

    global feat_model,model,tokenizer
    test_pic = test_pic.reshape((1,224,224,3))
    test_pic = preprocess_input(test_pic)
    test_pic = feat_model(test_pic)
    test_pic = np.reshape(test_pic,(4096,))
    test_pic = test_pic.reshape((1,-1))
    words = 'startseq'
    while 'endseq' not in words:
        test_seq = [words]
        test_seq = tokenizer.texts_to_sequences(test_seq)
        test_seq = np.array(test_seq)
        test_seq = pad_sequences(test_seq,maxlen=10)[0]
        test_seq = test_seq.reshape((1,10))

        pred = model([test_pic,test_seq])

        pred = get_word(np.argmax(pred))
        words = words+' '+pred
        
    words = words.replace('startseq','')
    words = words.replace('endseq','')
    return words.rstrip().lstrip()