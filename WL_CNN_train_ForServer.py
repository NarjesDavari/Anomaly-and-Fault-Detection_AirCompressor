# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 12:23:12 2022

@author: davar
"""


#import datetime as dt
import math
from skimage.transform import resize
import cv2  # for ingesting images
from scipy.stats import kurtosis, skew
from scipy.fftpack import fft
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
#import matplotlib
#import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')
# matplotlib.use('TkAgg')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#import datetime
import pywt
tf.compat.v1.random.set_random_seed(1)
#from scipy.signal import butter, filtfilt
#from time import time
#import seaborn as sns
print('OpenCV: %s' % cv2.__version__)  # print version
# %load_ext tensorboard
#import skimage.transform as st
#from IPython.display import display, HTML
#from tensorflow.keras.callbacks import ModelCheckpoint
#X_train_full =[]
#X_test_full = []
thr_boxplot = []

Bins = []
X_test_INPUT = []
rmse_OUTPUT = []
x = []
X_train = []
X_test = []


def load_Data_csvfile():
    # DataSet_trainig = pd.read_csv('C:/Users/davar/Documents/INESCTEC/code_python/Autoencoder/newData_21March2022/LargeDataForTraining_time.csv') #
    #DataSet_testAir = pd.read_csv('C:/Users/davar/Documents/INESCTEC/code_python/Autoencoder/newData_21March2022/MetroAir2_time.csv')
    #DataSet_testOil = pd.read_csv('C:/Users/davar/Documents/INESCTEC/code_python/Autoencoder/newData_21March2022/MetroOil2_time.csv')
    # DataSet_dump3 = pd.read_csv('C:/Users/davar/Documents/INESCTEC/code_python/Autoencoder/dump32020_time.csv')#newData_21March2022/MetroAirOil2_time
    # DataSet_dump4 = pd.read_csv('C:/Users/davar/Documents/INESCTEC/code_python/Autoencoder/dump42020_time.csv')#newData_21March2022/MetroAirOil2_time
    # DataSet_dump5 = pd.read_csv('C:/Users/davar/Documents/INESCTEC/code_python/Autoencoder/dump52020_time.csv')#newData_21March2022/MetroAirOil2_time
    #DataSet_training = pd.concat ((DataSet_dump3,DataSet_dump4,DataSet_dump5),ignore_index=True)
    DataSet = pd.read_csv("/home/ndavari/dataset_train.csv")
    columns_name = ['timestamp', 'TP2',	'TP3',	'H1', 'DV_pressure', 'Reservoirs',	'Oil_temperature',	'Flowmeter',	'Motor_current', 'COMP', 'DV_eletric',	'Towers',	'MPG',	'LPS',	'Pressure_switch',
                    'Oil_level','Caudal_impulses','timestampSecond']  # timestamp	TP2	TP3	H1	DV_pressure	Reservoirs	Oil_temperature	Flowmeter	Motor_current	COMP	DV_eletric	Towers	MPG	LPS	Pressure_switch	Oil_level	Caudal_impulses
    DataSet_trainig = DataSet[0:3000000]
    DataSet_Test = DataSet[3000000:]
    DataSet_trainig = DataSet_trainig[columns_name]
    DataSet_Test = DataSet_Test[columns_name]

    # if there is missing value: DataSet_Testfull.dropna(inplace=True, axis=1)
    return DataSet_trainig, DataSet_Test


def model_SimpleNetwork(input_dim, encoding_dim):
    # build model_0
    #input_dim = self.X_train.shape[1]
    input_layer = tf.keras.layers.Input(shape=(input_dim, ))
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(
        input_layer)  # ,activity_regularizer=keras.regularizers.l1(10e-5)
    # linear, softplus,tanh,  sigmoid( Sigmoid is a binary classifier, should not use it)
    decoded = tf.keras.layers.Dense(input_dim, activation='linear')(encoded)
    # map an input to its reconstruction
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.summary()
    return autoencoder


def model_DeepNetwor(input_dim, encoding_dim):
    # build model_1
    hidden_dim = int(encoding_dim / 2)
    learning_rate = 1e-3
    input_layer = tf.keras.layers.Input(shape=(input_dim, ))
    encoder = tf.keras.layers.Dense(encoding_dim, activation="relu",
                                    activity_regularizer=keras.regularizers.l1(learning_rate))(input_layer)
    encoder = tf.keras.layers.Dense(
        hidden_dim, activation="relu", activity_regularizer=keras.regularizers.l1(learning_rate))(encoder)
    encoder = tf.keras.layers.Dense(int(hidden_dim/2), activation="relu",
                                    activity_regularizer=keras.regularizers.l1(learning_rate))(encoder)
    encoder = tf.keras.layers.Dense(int(hidden_dim/5), activation='relu',
                                    activity_regularizer=keras.regularizers.l1(learning_rate))(encoder)
    decoder = tf.keras.layers.Dense(int(hidden_dim/2), activation="relu",
                                    activity_regularizer=keras.regularizers.l1(learning_rate))(encoder)
    decoder = tf.keras.layers.Dense(
        hidden_dim, activation="relu", activity_regularizer=keras.regularizers.l1(learning_rate))(encoder)
    decoder = tf.keras.layers.Dense(encoding_dim, activation="relu",
                                    activity_regularizer=keras.regularizers.l1(learning_rate))(decoder)
    decoder = tf.keras.layers.Dense(input_dim, activation="sigmoid")(
        decoder)  # activation="linear", softplus
    autoencoder = keras.Model(inputs=input_layer, outputs=decoder)
    autoencoder.summary()
    return autoencoder


def model_CNN_AE_1(X_train):

    input_img = tf.keras.layers.Input(
        shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))

    enc_conv1 = tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu', padding='same')(input_img)
    enc_pool1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(enc_conv1)
    enc_conv2 = tf.keras.layers.Conv2D(
        16, (4, 4), activation='relu', padding='same')(enc_pool1)
    enc_ouput = tf.keras.layers.MaxPooling2D((4, 4), padding='same')(enc_conv2)

    dec_conv2 = tf.keras.layers.Conv2D(
        16, (4, 4), activation='relu', padding='same')(enc_ouput)
    dec_upsample2 = tf.keras.layers.UpSampling2D((4, 4))(dec_conv2)
    dec_conv3 = tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu', padding='same')(dec_upsample2)
    dec_upsample3 = tf.keras.layers.UpSampling2D((2, 2))(dec_conv3)
    dec_output = tf.keras.layers.Conv2D(
        X_train.shape[3], (3, 3), activation='sigmoid', padding='same')(dec_upsample3)

    CNN_AE = keras.Model(input_img, dec_output)
    CNN_AE.summary()
    return CNN_AE


def model_CNN_AE_2(X_train):
    input = tf.keras.layers.Input(
        shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    # Encoder
    x = tf.keras.layers.Conv2D(64, kernel_size=(
        3, 3), activation="relu", padding="same")(input)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    # Default='valid', One of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
    x = tf.keras.layers.Conv2D(32, kernel_size=(
        3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    # Decoder
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=(
        3, 3), strides=2, activation="relu", padding="same")(x)
    # strides: Default=(1,1), An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(
        3, 3), strides=2, activation="relu", padding="same")(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    x = tf.keras.layers.Conv2D(X_train.shape[3], kernel_size=(
        3, 3), activation="sigmoid", padding="same")(x)
    # Autoencoder
    CNN_AE = tf.keras.models.Model(input, x)
    CNN_AE.summary()
    return CNN_AE


def model_CNN_AE_3(X_train):
    input_img = keras.Input(
        shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))

    x = tf.keras.layers.Conv2D(
        50, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = tf.keras.layers.Conv2D(
        8, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(
        50, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2D(
        X_train.shape[3], (3, 3), activation='sigmoid', padding='same')(x)
    CNN_AE = tf.keras.Model(input_img, decoded)
    CNN_AE.summary()
    return CNN_AE


def model_CNN_AE_4(X_train):
    input_img = keras.Input(
        shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))

    x = tf.keras.layers.Conv2D(
        100, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = tf.keras.layers.Conv2D(
        8, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(
        100, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2D(
        X_train.shape[3], (3, 3), activation='sigmoid', padding='same')(x)
    CNN_AE = tf.keras.Model(input_img, decoded)
    CNN_AE.summary()
    return CNN_AE


def cnn_model(X_train):  # used for classification
    model = tf.keras.models.Sequential()

# 2 Convolution layer with Max polling
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu",
              padding='same', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(
        5, 5), activation="relu", padding='same', kernel_initializer="he_normal"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(tf.keras.layers.Flatten())

# 3 Full connected layer
    model.add(tf.keras.layers.Dense(
        128, activation="relu", kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Dense(
        54, activation="relu", kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Dense(6, activation='softmax'))  # 6 classes

# summarize the model
    print(model.summary())
    return model


def simple_LPF(x, alpha):
    Y_LPF = np.zeros([len(x)])
    Y_LPF[0] = 1
    for i in range(1, len(x)):
        # if x[i]==0:
        Y_LPF[i] = Y_LPF[i-1]+alpha*(x[i]-Y_LPF[i-1])
        #  else:
        #  Y_LPF [i]=Y_LPF [i-1]+(1-alpha)*(x[i]-Y_LPF[i-1])
    pred_y_test_simpleLPF = Y_LPF
    return pred_y_test_simpleLPF


def Find_BestDuration(data, flag_training):
    COMP = data['COMP']
    Motor_current = data['Motor_current']
    TP3 = data['TP3']
    TP2 = data['TP2']
    H1 = data['H1']
    Flowmeter = data['Flowmeter']
    Oil_temp = data['Oil_temperature']
    DV_Pre = data['DV_pressure']
    Reservoirs = data['Reservoirs']
    #TP3 = 16 * (1 - ((20 - TP3) / 16))
    # timestampSecond
    data['timestamp'] = data['timestamp'].str.replace('\..*', '')
    DT = data['timestamp'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())
    # data['timestamp'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())  #.%f
    flag_LH = 0
    flag_HL = 0
    flag_interrupt_b = 0
    flag_interrupt_a = 0

    t_HL = []
    t_LH = []

    index_HL = []
    index_LH = []
    k = 2
    flag_start = 0
    flag_LH_after_b = 0
    flag_LH_after_a = 0
    flag_msg = 0
    num_bin_inc = 2
    num_bin_dec = 5
    Time_Start = []
    Time_End = []
    IndexEnd = []
    IndexStart = []
    dur_high_comp = []
    dur_low_comp = []
    Bins = []  # np.empty((0,5*7+3+2), float)
    T_start_msg = []
    T_end_msg = []
    work_time_more420 = []
    count_w_comp = 0
    while flag_start == 0:
        if COMP[k-1] == 1 and COMP[k] == 0 and TP3[k] > 8:
            t_HL_0 = DT[k]
            t_HL.append(t_HL_0)
            index_HL.append(k)
            flag_start = 1
        else:
            k = k+1
    for i in range(k+1, len(COMP)):

        # five min
        if flag_training == 1 and int(np.ceil(DT[i]-DT[i-1])) < 60*5 or flag_training == 0:
            if (COMP[i-1] == 0 and COMP[i] == 1):  # low to high
                t_LH.append(DT[i])
                flag_LH = 1
                index_LH.append(i)
            if (COMP[i-1] == 1 and COMP[i] == 0):  # high to low
                t_HL.append(DT[i])
                flag_HL = 1
                index_HL.append(i)

        if flag_HL == 1 and flag_LH == 1 and int(np.ceil(abs(t_LH[-1]-t_HL[-1]))) < 60:
            flag_HL = 0
            flag_LH = 0
            del t_LH[-1]
            del index_LH[-1]
            del t_HL[-1]
            del index_HL[-1]

        # and flag_training==1 case a
        if int(np.ceil(DT[i]-DT[i-1])) > 60*5 and flag_HL == 0 and flag_LH == 0:
            flag_interrupt_a = 1

        # and flag_training==1 case b
        if int(np.ceil(DT[i]-DT[i-1])) > 60*5 and flag_LH == 1 and flag_HL == 0:
            flag_LH = 0
            del t_LH[-1]
            del index_LH[-1]
            flag_interrupt_b = 1

        if flag_interrupt_b == 1 and flag_LH == 1 and flag_HL == 0:  # and flag_training==1:
            flag_LH = 0
            flag_interrupt_b = 0
            del t_LH[-1]
            del index_LH[-1]
            flag_LH_after_b = 1

        if flag_interrupt_b == 1 and flag_LH == 0 and flag_HL == 1:  # and flag_training==1:
            flag_HL = 0
            flag_interrupt_b = 0

        if flag_interrupt_a == 1 and flag_LH == 1 and flag_HL == 0:  # and flag_training==1:
            flag_LH = 0
            flag_interrupt_a = 0
            del t_LH[-1]
            del index_LH[-1]
            flag_LH_after_a = 1
        if flag_LH_after_a == 1 and flag_HL == 1:  # and flag_training==1:
            flag_LH_after_a = 0
            flag_HL = 0
        # and flag_training==1:
        if flag_LH_after_b == 1 and flag_interrupt_a == 1 and flag_HL == 0 and flag_LH == 0:
            flag_LH_after_b = 0

        if flag_interrupt_a == 1 and flag_HL == 1 and flag_LH == 0:  # and flag_training==1:
            if flag_interrupt_b == 1:
                flag_interrupt_b = 0
            flag_HL = 0
            flag_interrupt_a = 0

        # and flag_training==1:
        if flag_LH_after_b == 1 and flag_HL == 1 and flag_interrupt_a == 0:
            flag_LH_after_b = 0
            flag_HL = 0

        if flag_HL == 1 and flag_LH == 1:
            if flag_training == 1:
                # t_HL[-1]>t_LH[-1] and#and 60<(t_HL[len(t_HL)-1]-t_LH[len(t_LH)-1]):
                if int(np.ceil(t_HL[-1]-t_LH[-1])) < 1000 or int(np.ceil(t_HL[-1]-t_LH[-1])) > 3000:
                    flag_HL = 0
                    flag_LH = 0
                    del t_LH[-1]
                    del index_LH[-1]
            else:
                if t_HL[-1] > t_LH[-1] and int(np.ceil(t_HL[-1]-t_LH[-1])) < 120:
                    flag_HL = 0
                    flag_LH = 0
                    del t_LH[-1]
                    del index_LH[-1]
                    del t_HL[-1]
                    del index_HL[-1]

        if flag_msg == 0 and flag_LH == 0 and flag_HL == 0 and flag_training == 0 and COMP[i] == 0 and flag_interrupt_a == 0 and flag_interrupt_b == 0 and flag_LH_after_b == 0 and flag_LH_after_a == 0:
            count_w_comp = count_w_comp+1
            if count_w_comp > 500 and int(np.ceil(DT[i]-t_HL[-1])) > 420:
                print("Comp is Working more than",
                      int(np.ceil(DT[i]-t_HL[-1])))
                flag_msg = 1
                T_start_msg.append(DT[i])
        if flag_msg == 1 and flag_LH == 1 and flag_HL == 0 and flag_interrupt_a == 0:
            print("Comp Worked for more than", int(np.ceil(t_LH[-1]-t_HL[-1])))
            count_w_comp = 0
            flag_msg = 0
            work_time_more420.append(int(np.ceil(t_LH[-1]-t_HL[-1])))
            T_end_msg.append(DT[i])
        elif flag_msg == 1 and flag_LH == 0 and flag_HL == 0 and flag_interrupt_a == 1:
            print("Comp Worked for more than", int(np.ceil(DT[i]-t_HL[-1])))
            count_w_comp = 0
            flag_msg = 0
            work_time_more420.append(int(np.ceil(DT[i]-t_HL[-1])))
            T_end_msg.append(DT[i])

        if flag_HL == 1 and flag_LH == 1:

            if int(np.ceil(t_LH[-1]-t_HL[-2])) < 60:
                flag_HL = 0
                flag_LH = 0
                del t_LH[-1]
                del index_LH[-1]
                del t_HL[-2]
                del index_HL[-2]
            # elif flag_training==0 and t_LH[-1]> t_HL[-2] and (t_LH[-1]-t_HL[-2]).total_seconds()>420:
            #   flag_HL=0
             #  flag_LH=0
                # del t_LH[-1]
             #  del index_LH[-1]
            elif flag_training == 1 and t_LH[-1] > t_HL[-2] and int(np.ceil(t_LH[-1]-t_HL[-2])) > 420:
                flag_HL = 0
                flag_LH = 0
                del t_LH[-1]
                del index_LH[-1]
            else:
                #  data = [TP2,TP3,H1, Flowmeter, Motor_current]
                # for i in range(len(data)+1):
                #   B1,B2,B3,B4,B5,B6,B7= self.Extract_Bins(data[i:],index_LH,index_HL,t_HL,num_bin_inc,num_bin_dec)

                index11 = int((index_LH[-1]-index_HL[-2]) /
                              num_bin_inc + index_HL[-2])
                B1_tp2 = np.mean(
                    TP2[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])
                B1_tp3 = np.mean(
                    TP3[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])
                B1_H1 = np.mean(
                    H1[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])
                B1_FM = np.mean(
                    Flowmeter[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])
                B1_MC = np.mean(Motor_current[index_HL[len(
                    index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])  # .total_seconds()

                B2_tp2 = np.mean(TP2[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])
                B2_tp3 = np.mean(TP3[index11:index_LH[-1]]) * \
                    (t_HL[-1]-t_HL[-2])  # .total_seconds()
                B2_H1 = np.mean(H1[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])
                B2_FM = np.mean(
                    Flowmeter[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])
                # .total_seconds()
                B2_MC = np.mean(
                    Motor_current[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])

                index21 = int((index_HL[-1]-index_LH[-1]) /
                              num_bin_dec + index_LH[-1])
                B3_tp2 = np.mean(TP2[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])
                B3_tp3 = np.mean(TP3[index_LH[-1]:index21]) * \
                    (t_HL[-1]-t_HL[-2])  # .total_seconds()
                B3_H1 = np.mean(H1[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])
                B3_FM = np.mean(
                    Flowmeter[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])
                # .total_seconds()
                B3_MC = np.mean(
                    Motor_current[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])

                index22 = int(
                    2*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
                B4_tp2 = np.mean(TP2[index21:index22])*(t_HL[-1]-t_HL[-2])
                B4_tp3 = np.mean(TP3[index21:index22]) * \
                    (t_HL[-1]-t_HL[-2])  # .total_seconds()
                B4_H1 = np.mean(H1[index21:index22])*(t_HL[-1]-t_HL[-2])
                B4_FM = np.mean(Flowmeter[index21:index22])*(t_HL[-1]-t_HL[-2])
                # .total_seconds()
                B4_MC = np.mean(
                    Motor_current[index21:index22])*(t_HL[-1]-t_HL[-2])

                index23 = int(
                    3*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
                B5_tp2 = np.mean(TP2[index22:index23])*(t_HL[-1]-t_HL[-2])
                B5_tp3 = np.mean(TP3[index22:index23]) * \
                    (t_HL[-1]-t_HL[-2])  # .total_seconds()
                B5_H1 = np.mean(H1[index22:index23])*(t_HL[-1]-t_HL[-2])
                B5_FM = np.mean(Flowmeter[index22:index23])*(t_HL[-1]-t_HL[-2])
                # .total_seconds()
                B5_MC = np.mean(
                    Motor_current[index22:index23])*(t_HL[-1]-t_HL[-2])

                index24 = int(
                    4*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
                B6_tp2 = np.mean(TP2[index23:index24])*(t_HL[-1]-t_HL[-2])
                B6_tp3 = np.mean(TP3[index23:index24]) * \
                    (t_HL[-1]-t_HL[-2])  # .total_seconds()
                B6_H1 = np.mean(H1[index23:index24])*(t_HL[-1]-t_HL[-2])
                B6_FM = np.mean(Flowmeter[index23:index24])*(t_HL[-1]-t_HL[-2])
                # .total_seconds()
                B6_MC = np.mean(
                    Motor_current[index23:index24])*(t_HL[-1]-t_HL[-2])

                index25 = int(
                    5*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
                B7_tp2 = np.mean(TP2[index24:index25])*(t_HL[-1]-t_HL[-2])
                B7_tp3 = np.mean(TP3[index24:index25])*(t_HL[-1]-t_HL[-2])
                B7_H1 = np.mean(H1[index24:index25]) * \
                    (t_HL[-1]-t_HL[-2])  # .total_seconds()
                B7_FM = np.mean(Flowmeter[index24:index25])*(t_HL[-1]-t_HL[-2])
                # .total_seconds()
                B7_MC = np.mean(
                    Motor_current[index24:index25])*(t_HL[-1]-t_HL[-2])

                Time_Start.append(datetime.fromtimestamp(t_HL[-2]))
                Time_End.append(datetime.fromtimestamp(t_HL[-1]))
                # .total_seconds())
                dur_low_comp.append(int(np.ceil(t_LH[-1]-t_HL[-2])))
                # .total_seconds())
                dur_high_comp.append(int(np.ceil(t_HL[-1]-t_LH[-1])))
                IndexEnd.append(index_HL[-1])
                IndexStart.append(index_HL[-2])

                TL = (t_LH[-1]-t_HL[-2])  # .total_seconds()
                TH = (t_HL[-1]-t_LH[-1])  # .total_seconds()
                BIN = []
                # bbin=[]
                Bin_tp2 = [B1_tp2, B2_tp2, B3_tp2,
                           B4_tp2, B5_tp2, B6_tp2, B7_tp2]
                BIN.extend(Bin_tp2)
                # bbin.extend(Bin_tp2)
                Bin_tp3 = [B1_tp3, B2_tp3, B3_tp3,
                           B4_tp3, B5_tp3, B6_tp3, B7_tp3]
                BIN.extend(Bin_tp3)
                # bbin.extend(Bin_tp3)
                Bin_H1 = [B1_H1, B2_H1, B3_H1, B4_H1, B5_H1, B6_H1, B7_H1]
                BIN.extend(Bin_H1)
                Bin_FM = [B1_FM, B2_FM, B3_FM, B4_FM, B5_FM, B6_FM, B7_FM]
                BIN.extend(Bin_FM)
                Bin_MC = [B1_MC, B2_MC, B3_MC, B4_MC, B5_MC, B6_MC, B7_MC]
                BIN.extend(Bin_MC)

                Min_Oil = np.min(Oil_temp[index11:index25])
                Max_Oil = np.max(Oil_temp[index11:index25])
                MA1_Oil = np.mean(Oil_temp[index_HL[-2]:index_LH[-1]])
                MA2_Oil = np.mean(Oil_temp[index_LH[-1]:index_HL[-1]])
                Med_DV = np.median(DV_Pre[index11:index25])
                Med_Reser = np.median(Reservoirs[index11:index25])
                ss = [Min_Oil, Max_Oil, MA1_Oil,
                      MA2_Oil, Med_DV, Med_Reser, TL, TH]
                ss1 = [TL, TH]
                BIN.extend(ss)

                # bbin.extend(ss1)
                # self.Bins.append(bbin)
                Bins.append(BIN)
                flag_LH = 0
                flag_HL = 0

    return IndexStart, IndexEnd, Time_Start, Time_End


def Find_BestDuration_1(data, flag_training):
    COMP = data['COMP']
    Motor_current = data['Motor_current']
    TP3 = data['TP3']
    TP2 = data['TP2']
    H1 = data['H1']
    Flowmeter = data['Flowmeter']
    Oil_temp = data['Oil_temperature']
    DV_Pre = data['DV_pressure']
    Reservoirs = data['Reservoirs']
    #TP3 = 16 * (1 - ((20 - TP3) / 16))
    DT = data['timestampSecond']
    # data['timestamp'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())  #.%f
    flag_LH = 0
    flag_HL = 0
    flag_interrupt_b = 0
    flag_interrupt_a = 0

    t_HL = []
    t_LH = []

    index_HL = []
    index_LH = []
    k = 2
    flag_start = 0
    flag_LH_after_b = 0
    flag_LH_after_a = 0
    flag_msg = 0
    num_bin_inc = 2
    num_bin_dec = 5
    Time_Start = []
    Time_End = []
    dur_high_comp = []
    dur_low_comp = []
    IndexStart = []
    IndexEnd = []
    T_start_msg = []
    T_end_msg = []
    work_time_more420 = []
    count_w_comp = 0
    while flag_start == 0:
        if COMP[k-1] == 1 and COMP[k] == 0 and TP3[k] > 8:
            t_HL_0 = DT[k]
            t_HL.append(t_HL_0)
            index_HL.append(k)
            flag_start = 1
        else:
            k = k+1
    for i in range(k+1, len(COMP)):

        # five min
        if flag_training == 1 and int(np.ceil(DT[i]-DT[i-1])) < 60*5 or flag_training == 0:
            if (COMP[i-1] == 0 and COMP[i] == 1):  # low to high
                t_LH.append(DT[i])
                flag_LH = 1
                index_LH.append(i)
            if (COMP[i-1] == 1 and COMP[i] == 0):  # high to low
                t_HL.append(DT[i])
                flag_HL = 1
                index_HL.append(i)

        if flag_HL == 1 and flag_LH == 1 and int(np.ceil(abs(t_LH[-1]-t_HL[-1]))) < 60:
            flag_HL = 0
            flag_LH = 0
            del t_LH[-1]
            del index_LH[-1]
            del t_HL[-1]
            del index_HL[-1]

        if int(np.ceil(DT[i]-DT[i-1])) > 60*5 and flag_HL == 0 and flag_LH == 0 and flag_training == 1:  # case a
            flag_interrupt_a = 1

        if int(np.ceil(DT[i]-DT[i-1])) > 60*5 and flag_LH == 1 and flag_HL == 0 and flag_training == 1:  # case b
            flag_LH = 0
            del t_LH[-1]
            del index_LH[-1]
            flag_interrupt_b = 1

        if flag_interrupt_b == 1 and flag_LH == 1 and flag_HL == 0 and flag_training == 1:
            flag_LH = 0
            flag_interrupt_b = 0
            del t_LH[-1]
            del index_LH[-1]
            flag_LH_after_b = 1

        if flag_interrupt_b == 1 and flag_LH == 0 and flag_HL == 1 and flag_training == 1:
            flag_HL = 0
            flag_interrupt_b = 0

        if flag_interrupt_a == 1 and flag_LH == 1 and flag_HL == 0 and flag_training == 1:
            flag_LH = 0
            flag_interrupt_a = 0
            del t_LH[-1]
            del index_LH[-1]
            flag_LH_after_a = 1
        if flag_LH_after_a == 1 and flag_HL == 1 and flag_training == 1:
            flag_LH_after_a = 0
            flag_HL = 0
        if flag_LH_after_b == 1 and flag_interrupt_a == 1 and flag_HL == 0 and flag_LH == 0 and flag_training == 1:
            flag_LH_after_b = 0

        if flag_interrupt_a == 1 and flag_HL == 1 and flag_LH == 0 and flag_training == 1:
            if flag_interrupt_b == 1:
                flag_interrupt_b = 0
            flag_HL = 0
            flag_interrupt_a = 0

        if flag_LH_after_b == 1 and flag_HL == 1 and flag_interrupt_a == 0 and flag_training == 1:
            flag_LH_after_b = 0
            flag_HL = 0

        if flag_HL == 1 and flag_LH == 1:
            if flag_training == 1:
                # t_HL[-1]>t_LH[-1] and#and 60<(t_HL[len(t_HL)-1]-t_LH[len(t_LH)-1]):
                if int(np.ceil(t_HL[-1]-t_LH[-1])) < 1000 or int(np.ceil(t_HL[-1]-t_LH[-1])) > 3000:
                    flag_HL = 0
                    flag_LH = 0
                    del t_LH[-1]
                    del index_LH[-1]
            else:
                if t_HL[-1] > t_LH[-1] and int(np.ceil(t_HL[-1]-t_LH[-1])) < 120:
                    flag_HL = 0
                    flag_LH = 0
                    del t_LH[-1]
                    del index_LH[-1]
                    del t_HL[-1]
                    del index_HL[-1]

        if flag_msg == 0 and flag_LH == 0 and flag_HL == 0 and flag_training == 0 and COMP[i] == 0 and flag_interrupt_a == 0 and flag_interrupt_b == 0 and flag_LH_after_b == 0 and flag_LH_after_a == 0:
            count_w_comp = count_w_comp+1
            if count_w_comp > 500 and int(np.ceil(DT[i]-t_HL[-1])) > 420:
                print("Comp is Working more than",
                      int(np.ceil(DT[i]-t_HL[-1])))
                flag_msg = 1
                T_start_msg.append(DT[i])
        if flag_msg == 1 and flag_LH == 1 and flag_HL == 0 and flag_interrupt_a == 0:
            print("Comp Worked for more than", int(np.ceil(t_LH[-1]-t_HL[-1])))
            count_w_comp = 0
            flag_msg = 0
            work_time_more420.append(int(np.ceil(t_LH[-1]-t_HL[-1])))
            T_end_msg.append(DT[i])
        elif flag_msg == 1 and flag_LH == 0 and flag_HL == 0 and flag_interrupt_a == 1:
            print("Comp Worked for more than", int(np.ceil(DT[i]-t_HL[-1])))
            count_w_comp = 0
            flag_msg = 0
            work_time_more420.append(int(np.ceil(DT[i]-t_HL[-1])))
            T_end_msg.append(DT[i])

        if flag_HL == 1 and flag_LH == 1:

            if int(np.ceil(t_LH[-1]-t_HL[-2])) < 60:
                flag_HL = 0
                flag_LH = 0
                del t_LH[-1]
                del index_LH[-1]
                del t_HL[-2]
                del index_HL[-2]
            # elif flag_training==0 and t_LH[-1]> t_HL[-2] and (t_LH[-1]-t_HL[-2]).total_seconds()>420:
            #   flag_HL=0
             #  flag_LH=0
                # del t_LH[-1]
             #  del index_LH[-1]
            elif flag_training == 1 and t_LH[-1] > t_HL[-2] and int(np.ceil(t_LH[-1]-t_HL[-2])) > 420:
                flag_HL = 0
                flag_LH = 0
                del t_LH[-1]
                del index_LH[-1]
            else:
                #  data = [TP2,TP3,H1, Flowmeter, Motor_current]
                # for i in range(len(data)+1):
                #   B1,B2,B3,B4,B5,B6,B7= self.Extract_Bins(data[i:],index_LH,index_HL,t_HL,num_bin_inc,num_bin_dec)

                index11 = int((index_LH[-1]-index_HL[-2]) /
                              num_bin_inc + index_HL[-2])
                B1_tp2 = np.mean(
                    TP2[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])
                B1_tp3 = np.mean(
                    TP3[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])
                B1_H1 = np.mean(
                    H1[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])
                B1_FM = np.mean(
                    Flowmeter[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])
                B1_MC = np.mean(Motor_current[index_HL[len(
                    index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])  # .total_seconds()

                B2_tp2 = np.mean(TP2[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])
                B2_tp3 = np.mean(TP3[index11:index_LH[-1]]) * \
                    (t_HL[-1]-t_HL[-2])  # .total_seconds()
                B2_H1 = np.mean(H1[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])
                B2_FM = np.mean(
                    Flowmeter[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])
                # .total_seconds()
                B2_MC = np.mean(
                    Motor_current[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])

                index21 = int((index_HL[-1]-index_LH[-1]) /
                              num_bin_dec + index_LH[-1])
                B3_tp2 = np.mean(TP2[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])
                B3_tp3 = np.mean(TP3[index_LH[-1]:index21]) * \
                    (t_HL[-1]-t_HL[-2])  # .total_seconds()
                B3_H1 = np.mean(H1[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])
                B3_FM = np.mean(
                    Flowmeter[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])
                # .total_seconds()
                B3_MC = np.mean(
                    Motor_current[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])

                index22 = int(
                    2*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
                B4_tp2 = np.mean(TP2[index21:index22])*(t_HL[-1]-t_HL[-2])
                B4_tp3 = np.mean(TP3[index21:index22]) * \
                    (t_HL[-1]-t_HL[-2])  # .total_seconds()
                B4_H1 = np.mean(H1[index21:index22])*(t_HL[-1]-t_HL[-2])
                B4_FM = np.mean(Flowmeter[index21:index22])*(t_HL[-1]-t_HL[-2])
                # .total_seconds()
                B4_MC = np.mean(
                    Motor_current[index21:index22])*(t_HL[-1]-t_HL[-2])

                index23 = int(
                    3*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
                B5_tp2 = np.mean(TP2[index22:index23])*(t_HL[-1]-t_HL[-2])
                B5_tp3 = np.mean(TP3[index22:index23]) * \
                    (t_HL[-1]-t_HL[-2])  # .total_seconds()
                B5_H1 = np.mean(H1[index22:index23])*(t_HL[-1]-t_HL[-2])
                B5_FM = np.mean(Flowmeter[index22:index23])*(t_HL[-1]-t_HL[-2])
                # .total_seconds()
                B5_MC = np.mean(
                    Motor_current[index22:index23])*(t_HL[-1]-t_HL[-2])

                index24 = int(
                    4*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
                B6_tp2 = np.mean(TP2[index23:index24])*(t_HL[-1]-t_HL[-2])
                B6_tp3 = np.mean(TP3[index23:index24]) * \
                    (t_HL[-1]-t_HL[-2])  # .total_seconds()
                B6_H1 = np.mean(H1[index23:index24])*(t_HL[-1]-t_HL[-2])
                B6_FM = np.mean(Flowmeter[index23:index24])*(t_HL[-1]-t_HL[-2])
                # .total_seconds()
                B6_MC = np.mean(
                    Motor_current[index23:index24])*(t_HL[-1]-t_HL[-2])

                index25 = int(
                    5*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
                B7_tp2 = np.mean(TP2[index24:index25])*(t_HL[-1]-t_HL[-2])
                B7_tp3 = np.mean(TP3[index24:index25])*(t_HL[-1]-t_HL[-2])
                B7_H1 = np.mean(H1[index24:index25]) * \
                    (t_HL[-1]-t_HL[-2])  # .total_seconds()
                B7_FM = np.mean(Flowmeter[index24:index25])*(t_HL[-1]-t_HL[-2])
                # .total_seconds()
                B7_MC = np.mean(
                    Motor_current[index24:index25])*(t_HL[-1]-t_HL[-2])

                Time_Start.append(datetime.fromtimestamp(t_HL[-2]))
                Time_End.append(datetime.fromtimestamp(t_HL[-1]))
                # .total_seconds())
                dur_low_comp.append(int(np.ceil(t_LH[-1]-t_HL[-2])))
                # .total_seconds())
                dur_high_comp.append(int(np.ceil(t_HL[-1]-t_LH[-1])))
                IndexEnd.append(index_HL[-1])
                IndexStart.append(index_HL[-2])

                TL = (t_LH[-1]-t_HL[-2])  # .total_seconds()
                TH = (t_HL[-1]-t_LH[-1])  # .total_seconds()
                BIN = []
                Bin_tp2 = [B1_tp2, B2_tp2, B3_tp2,
                           B4_tp2, B5_tp2, B6_tp2, B7_tp2]
                BIN.extend(Bin_tp2)
                Bin_tp3 = [B1_tp3, B2_tp3, B3_tp3,
                           B4_tp3, B5_tp3, B6_tp3, B7_tp3]
                BIN.extend(Bin_tp3)
                Bin_H1 = [B1_H1, B2_H1, B3_H1, B4_H1, B5_H1, B6_H1, B7_H1]
                BIN.extend(Bin_H1)
                Bin_FM = [B1_FM, B2_FM, B3_FM, B4_FM, B5_FM, B6_FM, B7_FM]
                BIN.extend(Bin_FM)
                Bin_MC = [B1_MC, B2_MC, B3_MC, B4_MC, B5_MC, B6_MC, B7_MC]
                BIN.extend(Bin_MC)

                Med_Oil = np.median(Oil_temp[index11:index25])
                Med_DV = np.median(DV_Pre[index11:index25])
                Med_Reser = np.median(Reservoirs[index11:index25])
                ss = [Med_Oil, Med_DV, Med_Reser, TL, TH]
                BIN.extend(ss)

                Bins.append(BIN)
                # self.Bins.append(BIN)
                flag_LH = 0
                flag_HL = 0
    return IndexStart, IndexEnd, Time_Start, Time_End


def DigitalSensor_analysis(data, flag_training, IndexStart, IndexEnd):
    Xtrain = []
    Xtest = []
  #  self.ax1.set_title('Binary Code ' + str(single_date), fontsize=8)
    if len(IndexStart) > 0:
        mydict = {

        }
        for index in range(len(IndexStart)):
            #  aux = ''
            mydict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
            for j in range(IndexStart[index], IndexEnd[index]):
                aux = ''
                c = 0
                for valu in data.iloc[j, 9:].values:
                    if valu > 0:
                        # if c not in mydict.keys():
                        #   mydict[c] = 1
                        #  else:
                        mydict[c] += 1
                    c += 1

                # def keyfunction(k):
                  #  return mydict[k]

            x = []
            y = []
            # sorted(mydict, key=keyfunction, reverse=True)[:10]:
            for key in range(8):
                x.append(int(key))
                y.append(mydict[key])
             #   print("key: "+str(key)+" value: "+str(mydict[key]))

            # self.y.append(self.dur_low_comp[index])
            # self.y.append(self.dur_high_comp[index])
            if flag_training == 1:
                Xtrain.append(y)
            else:
                Xtest.append(y)
    return Xtrain, Xtest


def Add_Features(flag_training, X, IndexStart):

    if len(IndexStart) > 0:

        if flag_training == 1:
            X_train = Bins
            X_train = np.append(X_train, X, axis=1)
        else:
            X_test = Bins
            X_test = np.append(X_test, X, axis=1)


'''
    def scale_resize_image﻿(image)﻿:
        image = tf.image.convert_image_dtype(image, tf.float32) # equivalent to dividing image pixels by 255
        image = tf.image.resize(image, (﻿255﻿, 528﻿)﻿) # Resizing the image to 224x528 dimention
        return image
'''


def get_fft_values(tp3, FFT_flag, IndexStart, IndexEnd):
    def feature_generation_stochastic(x):
        Feature_oneWin = np.zeros(14)
        Mean_oneWin = np.mean(x, axis=0)
        Feature_oneWin[0] = Mean_oneWin
        # Feature_oneWin.append(Mean_oneWin)
        std_oneWin = np.std(x, axis=0)
        Feature_oneWin[1] = std_oneWin
        Skewness_oneWin = skew(x, bias=False, axis=0)
        Feature_oneWin[2] = Skewness_oneWin
        kurtosis_oneWin = kurtosis(x, bias=False, axis=0)
        Feature_oneWin[3] = kurtosis_oneWin
        # Quartil_oneWin=np.percentile(x, np.arange(0, 100, 25),axis=0) # quartiles
        # Feature_oneWin[4:8]=Quartil_oneWin
        Deciles_oneW = np.percentile(
            x, np.arange(0, 100, 10), axis=0)  # deciles
        Feature_oneWin[4:] = Deciles_oneW
        return Feature_oneWin

    def feature_generation_refPaper(x, freq):
        Feature_oneWin = np.zeros(10)
        L = len(x)
        Feature_oneWin[0] = np.mean(x, axis=0)
        Feature_oneWin[1] = sum(x*freq)/sum(x)
        Feature_oneWin[2] = np.sqrt(sum(x*(freq**2))/sum(x))
        Feature_oneWin[3] = np.sqrt(sum(x*(freq[0]-Feature_oneWin[1])**2)*L)
        Feature_oneWin[4] = np.sqrt(sum(x*freq**4)/sum(x*freq**2))
        Feature_oneWin[5] = np.sqrt(sum(x*freq**2)/(sum(x)*sum(x*freq**4)))
        Feature_oneWin[6] = Feature_oneWin[3] / Feature_oneWin[0]
        Feature_oneWin[7] = sum(
            x*(freq[0]-Feature_oneWin[0])**3)/L*Feature_oneWin[3]**3
        Feature_oneWin[8] = sum(
            x*(freq[0]-Feature_oneWin[4])**3)/L*Feature_oneWin[5]**4
        Feature_oneWin[9] = sum(
            x*np.sqrt(np.abs(freq[0]-Feature_oneWin[4])))/L*np.sqrt(Feature_oneWin[5])
        return Feature_oneWin

    dt = 1
    fs = 1 / dt
    Feature_list_FFT = []
    Feature_list_FFT_new = []
    for jj in range(len(IndexStart)):
        signal = tp3[IndexStart[jj]:IndexEnd[jj]]
        N = len(signal)
        if FFT_flag == 1:
            #f_values = np.linspace(0.0, 1.0 / (2.0 * dt), N // 2)
            fft_values_ = np.fft.fft(signal.tolist())
            fft_values = 2.0 / N * np.abs(fft_values_[1:N // 2])
            freq = np.fft.fftfreq(np.array(signal.tolist()).shape[-1])
            Feature = feature_generation_stochastic(fft_values)
            Feature_list_FFT.append(Feature)
            # new feature
            Feature_new = feature_generation_refPaper(
                fft_values, freq[1:N // 2])
            Feature_list_FFT_new.append(Feature_new)
        else:
            Feature = feature_generation_stochastic(signal.tolist())
            Feature_list_FFT.append(Feature)
    return Feature_list_FFT, Feature_list_FFT_new


def preprocess(array, sizeresize):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    Convert image data to numpy array and standardize values (divide by 255 since RGB values ranges from 0 to 255)
    """
    array = np.array(array, dtype="float") / 255.0
    print("Shape of whole data: ", array.shape)
    #array = array.astype("float32") / 255.0
    # array = np.reshape(array, (len(array), sizeresize, sizeresize, 1))
    return array


def creat_imag(X_input, scale, rescale_size, waveletname, sizeresize, IndexStart, IndexEnd):
    # image_resized=np.zeros((len(IndexStart),sizeresize,sizeresize))
    n_samples = len(IndexStart)
    n_signals = 1  # X_input.shape[1]
    scales = np.arange(1, scale + 1)
    # pre allocate array
    X_cwt = np.ndarray(shape=(n_samples, rescale_size,
                       rescale_size, n_signals), dtype='float32')

    for sample in range(n_samples):
        for sig in range(n_signals):
          #  signal = tp3[self.IndexStart[jj]:self.IndexEnd[jj]]
            signal = X_input[IndexStart[sample]:IndexEnd[sample]]  # , sig
            coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
            magn = (np.abs(coeff))
            rescale_coeffs = resize(magn, (rescale_size, rescale_size), mode='constant')
            
            X_cwt[sample, :, :, sig] = rescale_coeffs

    return X_cwt

def Standardization(X_train, X_test, flag_training):
    scaler = StandardScaler().fit(X_train)
    if flag_training == 1:
        # self.X_train_full.append(self.X_train)
        X_scaled = scaler.transform(X_train)
    else:
        # self.X_test_full.append(self.X_test)
        X_scaled = scaler.transform(X_test)
    return X_scaled

def Standardization_2(X):
    X_normalized = [(X[i,:,:,0]-np.min(X[i,:,:,0]))/(np.max(X[i,:,:,0])-np.min(X[i,:,:,0])) for i in range(np.shape(X)[0])]
    return X_normalized

def evalution(X_train_full, flag_training,i):
    # scaler = StandardScaler().fit(X_train)
    #X_train_scaled_full = scaler.transform(X_train)
    #X_train_scaled= Standardization(X_train,X_test,flag_training)

    # scaler = StandardScaler()
    # self.X_train_scaled = np.array(self.X_train).reshape(-1,1)
    # self.X_train_scaled = scaler.fit_transform(self.X_train_scaled)
    # self.X_train = self.X_train.flatten()
    #X_test = scaler.transform(X_test)
    ############################
    X_train_full=np.array(Standardization_2(X_train_full))
    X_train_full=X_train_full.reshape(X_train_full.shape[0],X_train_full.shape[1],X_train_full.shape[2],1)
    CNN_AE = model_CNN_AE_3(X_train_full)
    # CNN_AE =model_CNN_AE_1(X_train_full)
    #CNN = cnn_model(X_train_full)
    #autoencoder =model_DeepNetwor(input_dim,encoding_dim)
    #############################
    X_train, X_valid = train_test_split(
        X_train_full, test_size=0.2, shuffle=False)
    # shape of data, which is [samples, rows, columns, channels]
    print("Shape of X_train: ", X_train.shape)
    print("Shape of X_valid: ", X_valid.shape)
    ##########################
    opt=keras.optimizers.Adam()
    # learning_rate=0.001, momentums = [0.0, 0.5, 0.9, 0.99]
    #opt = keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, decay=0.001)
    # CNN_AE.compile(optimizer=opt, metrics=['accuracy'],loss='binary_crossentropy')#loss='SparseCategoricalCrossentropy',loss_weights=None,weighted_metrics=None, run_eagerly=None
    # , metrics=['accuracy']
    CNN_AE.compile(optimizer=opt, loss='mse',metrics=["accuracy"])
    file_name = 'my_saved_model_new'
    tensorboard_callback = TensorBoard(log_dir="logs\\{}".format(file_name))

    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # tensorBoard --logdir=logs/

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                     factor=0.5,
                                                     patience=20,
                                                     verbose=1,
                                                     min_delta=0.0001,
                                                     min_lr=1e-6,
                                                     mode='auto',
                                                     cooldown=5)
    EarlyStopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, mode="min")
    history = CNN_AE.fit(X_train, X_train, epochs=100, batch_size=50,
                         # validation_split=0.1,
                         shuffle=False,
                         # shuffle=True,
                         validation_data=(X_valid, X_valid),
                         callbacks=[EarlyStopping_callback, reduce_lr])  # ,tensorboard_callback
    # self.loss_training = history.history["loss"]

    # tf.summary.scaler
    # tf.summary.histogram
    # tf.summary.tensor

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Number of epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # plt.subplot(1,2,2)
    # plt.plot(history.history["accuracy"], label="Training accuracy")
    # plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    # plt.xlabel("Number of epoch")
    # plt.ylabel("accuracy")
    # plt.legend()
    # plt.show()

   
    Data1 = pd.DataFrame(history.history["val_loss"])
    Data1.to_csv(r'/home/ndavari/Result_WTCNN_Train/LossValidation_WLCNN_Train'+ str(i) + '.csv')
    
    Data1 = pd.DataFrame(history.history["accuracy"])
    Data1.to_csv(r'/home/ndavari/Result_WTCNN_Train/accuracyTraining_WLCNN_Train'+ str(i) + '.csv')

    Data1 = pd.DataFrame(history.history["val_accuracy"])
    Data1.to_csv(r'/home/ndavari/Result_WTCNN_Train/accuracyValidation_WLCNN_Train'+ str(i) + '.csv')

    #X_train_pre_resh = np.reshape(X_train_pred, (len(X_train_pred), np.shape(X_train_pred)[1], np.shape(X_train_pred)[2]))
    # X_train_resh = np.reshape(X_train, (len(X_train), np.shape(X_train)[1], np.shape(X_train)[2]))
    X_train_pred = CNN_AE.predict(X_train)
    train_rmse = np.ndarray(
        shape=(np.shape(X_train)[0], np.shape(X_train)[3]), dtype='float32')
    mean_error_train = np.ndarray(
        shape=(np.shape(X_train)[0], np.shape(X_train)[3]), dtype='float32')
    std_error_train = np.ndarray(
        shape=(np.shape(X_train)[0], np.shape(X_train)[3]), dtype='float32')
    for n_sig in range(np.shape(X_train)[3]):
        mean_error_train[:, n_sig] = [np.mean(
            X_train[i, :, :, n_sig]-X_train_pred[i, :, :, n_sig]) for i in range(np.shape(X_train)[0])]
        std_error_train[:, n_sig] = [np.std(
            X_train[i, :, :, n_sig]-X_train_pred[i, :, :, n_sig]) for i in range(np.shape(X_train)[0])]
        train_rmse[:, n_sig] = [mean_squared_error(
            X_train[i, :, :, n_sig], X_train_pred[i, :, :, n_sig], multioutput='uniform_average') for i in range(np.shape(X_train)[0])]
    #self.train_rmse_loss = np.sqrt(np.mean((X_train_pred - self.X_train)**2, axis=1))
    train_rmse_loss = train_rmse
    Data1 = pd.DataFrame(train_rmse)
    Data1.to_csv(r'/home/ndavari/Result_WTCNN_Train/LossTraining_WLCNN_Train'+ str(i) + '.csv')

    ### ma['EMA'] = ma['rms_t'].ewm(span=30,adjust=False).mean()
    plt.plot(train_rmse_loss)
    plt.xlabel("Number of X_train")
    plt.ylabel("train_rmse")
    plt.legend()
    plt.show()
    plt.hist(train_rmse_loss, bins=50)
    threshold = np.mean(train_rmse_loss)+3*np.std(train_rmse_loss)
    median = np.sqrt(2*math.log(len(train_rmse_loss)) /
                     len(train_rmse_loss))*(np.median(train_rmse_loss))/0.6745
    upper_quartile = np.percentile(train_rmse_loss, 75)
    lower_quartile = np.percentile(train_rmse_loss, 25)
    iqr = upper_quartile - lower_quartile
    # upper_whisker =self. train_rmse_loss[self.train_rmse_loss<=upper_quartile+1.5*iqr].max()
    # lower_whisker = self.train_rmse_loss[self.train_rmse_loss>=lower_quartile-1.5*iqr].min()
    thr_boxplot.append(upper_quartile+3*iqr)  # extreme outliers
    #######################################################
    X_valid_pred = CNN_AE.predict(X_valid)
    #X_valid_pre_resh = np.reshape(X_valid_pred, (len(X_valid_pred), np.shape(X_valid_pred)[1], np.shape(X_valid_pred)[2]))
    #X_valid_resh = np.reshape(X_valid, (len(X_valid), np.shape(X_valid)[1], np.shape(X_valid)[2]))
    valid_rmse = np.ndarray(
        shape=(np.shape(X_valid)[0], np.shape(X_valid)[3]), dtype='float32')
    for n_sig in range(np.shape(X_valid)[3]):
        valid_rmse[:, n_sig] = [mean_squared_error(
            X_valid[i, :, :, n_sig], X_valid_pred[i, :, :, n_sig], multioutput='uniform_average') for i in range(np.shape(X_valid)[0])]
    valid_rmse_loss = np.sqrt(valid_rmse)
    plt.hist(valid_rmse_loss, bins=50)
    threshold = np.max(valid_rmse_loss)
    median = np.median(valid_rmse_loss)
    upper_quartile = np.percentile(valid_rmse_loss, 75)
    lower_quartile = np.percentile(valid_rmse_loss, 25)
    iqr = upper_quartile - lower_quartile
    # upper_whisker =self. train_rmse_loss[self.train_rmse_loss<=upper_quartile+1.5*iqr].max()
    # lower_whisker = self.train_rmse_loss[self.train_rmse_loss>=lower_quartile-1.5*iqr].min()
    thr_boxplot.append(upper_quartile+3*iqr)  # extreme outliers

    return CNN_AE, thr_boxplot, train_rmse_loss


def evalution_TestData(CNN_AE, X_test, thr_boxplot, flag_training):
    # Time_StartCOMP=[]
    # Time_EndCOMP=[]
    # Y_Test_pre=[]
    # Y_Test_preLPF=[]

    X_test=np.array(Standardization_2(X_test))
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
    X_test_pred = CNN_AE.predict(X_test, verbose=0)
    #X_pre_resh = np.reshape(X_test_pred, (len(X_test_pred), np.shape(X_test_pred)[1], np.shape(X_test_pred)[2]))
    # X_test_resh = np.reshape(X_test, (len(X_test), np.shape(X_test)[1], np.shape(X_test)[2]))
    test_rmse = np.ndarray(
        shape=(np.shape(X_test)[0], np.shape(X_test)[3]), dtype='float32')
    for n_sig in range(np.shape(X_test)[3]):
        test_rmse[:, n_sig] = [mean_squared_error(
            X_test[i, :, :, n_sig], X_test_pred[i, :, :, n_sig], multioutput='uniform_average') for i in range(np.shape(X_test)[0])]
    test_rmse_loss = np.sqrt(test_rmse)
    plt.hist(test_rmse_loss, bins=50)
    pred_y_test = [1 if e > thr_boxplot[-1] else 0 for e in test_rmse_loss]
    alpha_SLPF = 0.07
    pred_y_test_simpleLPF = simple_LPF(pred_y_test, alpha_SLPF)
    
    return test_rmse_loss, pred_y_test,pred_y_test_simpleLPF

    


def prprocessing():
    flag_training = 1
    train_data, test_data = load_Data_csvfile()
    IndexStart_train, IndexEnd_train, Time_Start_train, Time_End_train = Find_BestDuration(
        train_data, flag_training)
    scale = 64  #six level decomposition
    # amount of pixels in X and Y
    #128(2^7) scales in seventh level of decomposition
    rescale_size = 128 
    sizeresize = 64
    waveletname = 'morl'
    tp3 = train_data['TP3']
    tp2 = train_data['TP2']
    COMP = train_data['COMP']
    MC = train_data['Motor_current']
    Oil = train_data['Oil_temperature']
    DV_Pre = train_data['DV_pressure']
    #X_CWT_train = np.transpose(np.array([tp2, tp3, COMP]))

    image_creat_tp2 = creat_imag(tp2, scale, rescale_size, waveletname, sizeresize, IndexStart_train, IndexEnd_train)
    image_creat_tp3 = creat_imag(tp3, scale, rescale_size, waveletname, sizeresize, IndexStart_train, IndexEnd_train)
    image_creat_MC = creat_imag(MC, scale, rescale_size, waveletname, sizeresize, IndexStart_train, IndexEnd_train)
    image_creat_Oil = creat_imag(Oil, scale, rescale_size, waveletname, sizeresize, IndexStart_train, IndexEnd_train)
    image_creat_DV = creat_imag(DV_Pre, scale, rescale_size, waveletname, sizeresize, IndexStart_train, IndexEnd_train)

    # Test
    flag_training = 0
    test_data = test_data.reset_index(drop=True)
    IndexStart_test, IndexEnd_test, Time_Start_test, Time_End_test = Find_BestDuration(
        test_data, flag_training)
    tp3_test = test_data['TP3']
    tp2_test = test_data['TP2']
    COMP_test = test_data['COMP']
    MC_test = test_data['Motor_current']
    Oil_test = test_data['Oil_temperature']
    DV_Pre_test = test_data['DV_pressure']
    #X_CWT_test = np.transpose(np.array([tp2_test, tp3_test, COMP_test]))
    image_creat_test_tp2 = creat_imag(tp2_test, scale, rescale_size, waveletname, sizeresize, IndexStart_test, IndexEnd_test)
    image_creat_test_tp3 = creat_imag(tp3_test, scale, rescale_size, waveletname, sizeresize, IndexStart_test, IndexEnd_test)
    image_creat_test_MC = creat_imag(MC_test, scale, rescale_size, waveletname, sizeresize, IndexStart_test, IndexEnd_test)
    image_creat_test_Oil = creat_imag(Oil_test, scale, rescale_size, waveletname, sizeresize, IndexStart_test, IndexEnd_test)
    image_creat_test_DV = creat_imag(DV_Pre_test, scale, rescale_size, waveletname, sizeresize, IndexStart_test, IndexEnd_test)

    Data2 = pd.DataFrame(Time_Start_test)
    Data2.to_csv(r'/home/ndavari/Result_WTCNN_Train/Time_Start_WLCNN_Train_test.csv')
    return image_creat_tp2,image_creat_tp3,image_creat_MC,image_creat_Oil,image_creat_DV,image_creat_test_tp2,image_creat_test_tp3,image_creat_test_MC,image_creat_test_Oil,image_creat_test_DV



def Run_network(X_train, X_test,i):

    flag_training = 1
    CNN_AE, thr_boxplot, train_rmse_loss = evalution(X_train, flag_training,i)
    flag_training = 0
    test_rmse_loss, pred_y_test,pred_y_test_simpleLPF = evalution_TestData(CNN_AE, X_test, thr_boxplot, flag_training)
    return train_rmse_loss,test_rmse_loss,pred_y_test,pred_y_test_simpleLPF,thr_boxplot



image_creat_tp2,image_creat_tp3,image_creat_MC,image_creat_Oil,image_creat_DV,image_creat_test_tp2,image_creat_test_tp3,image_creat_test_MC,image_creat_test_Oil,image_creat_test_DV = prprocessing()
X_train=[]
X_train.append(image_creat_tp2)
X_train.append(image_creat_tp3)
X_train.append(image_creat_MC)
X_train.append(image_creat_Oil)
X_train.append(image_creat_DV)
X_test=[]
X_test.append(image_creat_test_tp2)
X_test.append(image_creat_test_tp3)
X_test.append(image_creat_test_MC)
X_test.append(image_creat_test_Oil)
X_test.append(image_creat_test_DV)
#train_rmse_loss=[]
#test_rmse_loss=[]
#pred_y_test=[]
#pred_y_test_simpleLPF=[]
for i in range(5):
    train_rmse_loss,test_rmse_loss,pred_y_test,pred_y_test_simpleLPF,thr_boxplot=Run_network(X_train[i], X_test[i],i)
    Data1 = pd.DataFrame(np.array(train_rmse_loss))
    Data1.to_csv(r'/home/ndavari/Result_WTCNN_Train/train_rmse_loss_WLCNN_Train'+ str(i) + '.csv')
    Data2 = pd.DataFrame(np.array(test_rmse_loss))
    Data2.to_csv(r'/home/ndavari/Result_WTCNN_Train/RMSError(OUTPUT)_WLCNN_Train'+ str(i) + '.csv')
    Data3 = pd.DataFrame(pred_y_test)
    Data3.to_csv(r'/home/ndavari/Result_WTCNN_Train/Y_testPredict_WLCNN_Train'+ str(i) + '.csv')
    Data4 = pd.DataFrame(pred_y_test_simpleLPF)
    Data4.to_csv(r'/home/ndavari/Result_WTCNN_Train/Y_testPredictLPF_WLCNN_Traing'+ str(i) + '.csv')
    Data5 = pd.DataFrame(thr_boxplot)
    Data5.to_csv(r'/home/ndavari/Result_WTCNN_Train/Threshold_Boxplot_WLCNN_Train'+ str(i) + '.csv')
    print("******************")
    print("Analysis TrainData #",i)
    print("******************")
