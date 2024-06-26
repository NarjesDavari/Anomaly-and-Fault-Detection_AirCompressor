# -*- coding: utf-8 -*-
"""
Created on Sun May 16 22:45:44 2021

@author: davari
"""

import datetime as dt
import numpy as np
import mysql.connector
import pandas as pd
from scipy import integrate
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Qt5Agg')  
#matplotlib.use('TkAgg')
import matplotlib.animation as animation
from scipy.fftpack import fft
from scipy import signal
from sklearn.preprocessing import StandardScaler,RobustScaler,PowerTransformer,MinMaxScaler,MaxAbsScaler,Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.metrics import auc, roc_curve, mean_squared_error
from matplotlib.patches import Rectangle
import pywt
import datetime
tf.compat.v1.random.set_random_seed(1)
from scipy.signal import butter, filtfilt
import math



class ZombieScript:

    def load_Data(self, startDay,endDay, train):
        print(startDay.strftime('%d-%m-%Y %H:%M:%S'))
        try:
            mylocaldb = mysql.connector.connect(
                host="194.117.29.209",
                database='fstopper',
                user="narges",
                passwd="narges"
            )
        except mysql.connector.Error as err:
            print(err)
        try:
            mycursor = mylocaldb.cursor()
            #nextDay = startDay + dt.timedelta(hours=1)
            mycursor.execute(
                "SELECT readtimestamp, anch2, anch8, di1 FROM trainsmp where train=" +
                str(train) + " AND readtimestamp between \"" +
                startDay.strftime('%Y-%m-%d %H:%M:%S') + "\" AND \"" + endDay.strftime(
                    '%Y-%m-%d %H:%M:%S') + "\";")
            myresult = mycursor.fetchall()
            columns_name = ['timestamp', 'TP3', 'MC','di1']
            self.data = pd.DataFrame(data=list(myresult), columns=columns_name)
            mycursor.close()
        except mysql.connector.Error as err:
            print(err)


    
    def model_SimpleNetwork(self,input_dim,encoding_dim):
        ##build model_0
        #input_dim = self.X_train.shape[1] 
        input_layer = tf.keras.layers.Input(shape=(input_dim, )) 
        encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer) #,activity_regularizer=keras.regularizers.l1(10e-5)
        decoded = tf.keras.layers.Dense(input_dim, activation='linear')(encoded)#linear, softplus,tanh,  sigmoid( Sigmoid is a binary classifier, should not use it)
        self.autoencoder = keras.Model(input_layer, decoded) # map an input to its reconstruction
        self.autoencoder.summary()
      
    def model_AE_CNN(self,input_dim):
        input_img = keras.Input(shape=(input_dim, input_dim, 1))

        x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
        #x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
       # x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(x)
       #x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(x)
       #x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        encoded = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(encoded)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        decoded = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

        self.AE_CNN = keras.Model(input_img, decoded)
        self.AE_CNN.summary()

        
        
    
    def simple_LPF(self,x,alpha):
        Y_LPF=np.zeros([len(x)])
        Y_LPF[0]=1
        for i in range(1,len(x)):
            #if x[i]==0:
            Y_LPF [i]=Y_LPF [i-1]+alpha*(x[i]-Y_LPF[i-1])
            #  else:
                #  Y_LPF [i]=Y_LPF [i-1]+(1-alpha)*(x[i]-Y_LPF[i-1])
        self.pred_y_test_simpleLPF=Y_LPF


    def Find_BestDuration(self,flag_training, start_date,end_date,train):   
        
        self.load_Data(start_date,end_date, train)
        COMP = self.data.iloc[:,3] 
        Motor_current = self.data.iloc[:,2] 
        TP3 = self.data.iloc[:,1] 
        TP3 = 16 * (1 - ((20 - TP3) / 16))
        DT = self.data.iloc[:,0]
        flag_LH =0
        flag_HL =0
        flag_interrupt_b =0
        flag_interrupt_a =0
    
        t_HL=[]
        t_LH=[]
   
        index_HL=[]
        index_LH=[]
        k=1
        flag_start=0
        flag_LH_after_b=0
        flag_LH_after_a=0
        flag_msg=0
        num_bin_inc=10
        num_bin_dec=100
        self.Time_Start =[]
        self.Time_end =[]
        self.IndexEnd=[]
        self.IndexStart=[]
        self.dur_high_comp=[]
        self.dur_low_comp=[]
        self.Bins =np.empty((0,num_bin_inc+num_bin_dec), float)
       # self.Bins_full=np.empty((0,2*(num_bin_inc+num_bin_dec)+2), float)
        T_start_msg=[]
        T_end_msg=[]
        work_time_more420=[]
        count_w_comp=0
        while flag_start==0:
            if COMP[k-1]==1 and COMP[k]==0 and TP3[k]>8:
                t_HL_0 =DT[k]
                t_HL.append(t_HL_0)
                index_HL.append(k)
                flag_start=1
            else:
                k=k+1
        for i in range(k+1,len(COMP)):#

            if flag_training==1 and (DT[i]-DT[i-1]).total_seconds()<60*5 or flag_training==0: #five min 
               if (COMP[i-1]==0 and COMP[i]==1): #low to high
                  t_LH.append(DT[i])
                  flag_LH =1
                  index_LH.append(i) 
               if (COMP[i-1]==1 and COMP[i]==0):#high to low
                  t_HL.append(DT[i])
                  flag_HL = 1
                  index_HL.append(i)
   
            if flag_HL ==1 and flag_LH ==1 and abs(t_LH[-1]-t_HL[-1]).total_seconds()<60:
                flag_HL =0
                flag_LH =0
                del t_LH[-1]
                del index_LH[-1]
                del t_HL[-1]
                del index_HL[-1]
            
            if (DT[i]-DT[i-1]).total_seconds()>60*5 and flag_HL ==0 and flag_LH==0 and flag_training==1:  #case a
                flag_interrupt_a =1
     
            if (DT[i]-DT[i-1]).total_seconds()>60*5 and flag_LH ==1 and flag_HL ==0 and flag_training==1:  #case b
                flag_LH=0
                del t_LH[-1]
                del index_LH[-1]
                flag_interrupt_b =1
       
            if flag_interrupt_b ==1 and flag_LH ==1 and flag_HL==0 and flag_training==1:
                flag_LH =0
                flag_interrupt_b =0
                del t_LH[-1]
                del index_LH[-1]
                flag_LH_after_b =1

      
            if flag_interrupt_b ==1 and flag_LH==0 and flag_HL ==1 and flag_training==1:
                flag_HL =0
                flag_interrupt_b =0
    
            if flag_interrupt_a ==1 and flag_LH==1 and flag_HL==0 and flag_training==1:
                flag_LH=0
                flag_interrupt_a =0
                del t_LH[-1]
                del index_LH[-1]
                flag_LH_after_a =1
            if flag_LH_after_a ==1 and flag_HL==1 and flag_training==1:
                flag_LH_after_a=0
                flag_HL=0
            if flag_LH_after_b==1 and flag_interrupt_a ==1 and flag_HL==0 and flag_LH==0 and flag_training==1:
                flag_LH_after_b=0
        
            if flag_interrupt_a ==1 and flag_HL==1 and flag_LH==0 and flag_training==1:
                if flag_interrupt_b==1:
                    flag_interrupt_b=0
                flag_HL=0
                flag_interrupt_a =0
        
            if flag_LH_after_b==1 and flag_HL==1 and flag_interrupt_a==0 and flag_training==1:
                flag_LH_after_b=0 
                flag_HL=0 
       
            if flag_HL==1 and flag_LH==1: 
                if flag_training==1:
                   if  (t_HL[-1]-t_LH[-1]).total_seconds()<1000 or (t_HL[-1]-t_LH[-1]).total_seconds()>3000: ## t_HL[-1]>t_LH[-1] and#and 60<(t_HL[len(t_HL)-1]-t_LH[len(t_LH)-1]): 
                       flag_HL=0
                       flag_LH=0
                       del t_LH[-1]
                       del index_LH[-1] 
                else:
                    if t_HL[-1]>t_LH[-1] and (t_HL[-1]-t_LH[-1]).total_seconds()<120:
                        flag_HL=0
                        flag_LH=0
                        del t_LH[-1]
                        del index_LH[-1]
                        del t_HL[-1]
                        del index_HL[-1]

            if flag_msg==0 and flag_LH==0 and flag_HL==0 and flag_training==0 and COMP[i]==0 and flag_interrupt_a==0 and flag_interrupt_b==0 and flag_LH_after_b==0 and flag_LH_after_a==0: 
               count_w_comp = count_w_comp+1
               if count_w_comp>500 and (DT[i]-t_HL[-1]).total_seconds()>420:
                   print("Comp is Working more than",(DT[i]-t_HL[-1]).total_seconds())
                   flag_msg = 1
                   T_start_msg.append(DT[i])
            if flag_msg==1 and flag_LH==1 and flag_HL==0 and flag_interrupt_a==0:
                print("Comp Worked for more than", (t_LH[-1]-t_HL[-1]).total_seconds())
                count_w_comp=0
                flag_msg=0
                work_time_more420.append((t_LH[-1]-t_HL[-1]).total_seconds())
                T_end_msg.append(DT[i])
            elif flag_msg==1 and flag_LH==0 and flag_HL==0 and flag_interrupt_a==1:
                print("Comp Worked for more than", (DT[i]-t_HL[-1]).total_seconds())
                count_w_comp=0
                flag_msg=0
                work_time_more420.append((DT[i]-t_HL[-1]).total_seconds())
                T_end_msg.append(DT[i])
           
            if flag_HL==1 and flag_LH==1: 

                if (t_LH[-1]-t_HL[-2]).total_seconds()<60:
                    flag_HL=0
                    flag_LH=0
                    del t_LH[-1]
                    del index_LH[-1] 
                    del t_HL[-2]
                    del index_HL[-2]
                #elif flag_training==0 and t_LH[-1]> t_HL[-2] and (t_LH[-1]-t_HL[-2]).total_seconds()>420:
                #   flag_HL=0
                 #  flag_LH=0
                  # del t_LH[-1]
                 #  del index_LH[-1]
                elif flag_training==1 and t_LH[-1]> t_HL[-2] and (t_LH[-1]-t_HL[-2]).total_seconds()>420:
                   flag_HL=0
                   flag_LH=0
                   del t_LH[-1]
                   del index_LH[-1]     
                else:
                     
                     BinTP3_inc=[0]*num_bin_inc #np.zeros(num_bin_inc)
                     BinMC_inc=[0]*num_bin_inc #np.zeros(num_bin_inc)
                     index_inc_s=[0]*(num_bin_inc+1) #np.zeros(num_bin_inc+1, dtype=int)
                     index_inc_e=[0]*num_bin_inc #np.zeros(num_bin_inc, dtype=int)
                     index_inc_s[0]=int(index_HL[-2])
                     for j in range(num_bin_inc):
                         index_inc_e[j] = int((j+1)*(index_LH[-1]-index_HL[-2])/num_bin_inc + index_HL[-2])
                         BinTP3_inc[j]=np.mean(TP3[index_inc_s[j]:index_inc_e[j]])*(t_HL[-1]-t_HL[-2]).total_seconds()
                         BinMC_inc[j]=np.mean(Motor_current[index_inc_s[j]:index_inc_e[j]])*(t_HL[-1]-t_HL[-2]).total_seconds()
                         index_inc_s[j+1]=  index_inc_e[j] 
                         
                     index_dec_s=[0]*(num_bin_dec+1)  #np.zeros(num_bin_dec+1, dtype=int)
                     index_dec_e=[0]*num_bin_dec #np.zeros(num_bin_dec, dtype=int)
                     BinTP3_dec=[0]*num_bin_dec #np.zeros(num_bin_dec)
                     BinMC_dec=[0]*num_bin_dec #np.zeros(num_bin_dec)
                     index_dec_s[0]= int(index_LH[-1])
                     for j in range(num_bin_dec):
                         index_dec_e[j] = int((j+1)*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
                         BinTP3_dec[j]=np.mean(TP3[index_dec_s[j]:index_dec_e[j]])*(t_HL[-1]-t_HL[-2]).total_seconds()
                         BinMC_dec[j]=np.mean(Motor_current[index_dec_s[j]:index_dec_e[j]])*(t_HL[-1]-t_HL[-2]).total_seconds()
                         index_dec_s[j+1]=  index_dec_e[j] 
                                
              
                     self.Time_Start.append(t_HL[-2])
                     self.Time_end.append(t_HL[-1])
                     self.dur_low_comp.append((t_LH[-1]-t_HL[-2]).total_seconds())
                     self.dur_high_comp.append((t_HL[-1]-t_LH[-1]).total_seconds())
                     self.IndexEnd.append(index_HL[-1])
                     self.IndexStart.append(index_HL[-2])
                     
                     TL=(t_LH[-1]-t_HL[-2]).total_seconds()
                     TH=(t_HL[-1]-t_LH[-1]).total_seconds()
                     TLH=[TL,TH]
                     
                     BIN=np.array([BinTP3_inc+BinTP3_dec])
                     self.Bins = np.append(self.Bins, BIN, axis=0)
                     
                    # BIN_full=np.array([BinTP3_inc+BinMC_inc+BinMC_dec+BinTP3_dec+TLH])
                  #   self.Bins_full = np.append(self.Bins_full, BIN_full, axis=0)
                     flag_LH=0
                     flag_HL=0          


    def ConvertSignaltoIMAGE(self,flag_training,scales,waveletname):

        if len(self.IndexStart)>0:
            Coeff=np.zeros((len(self.Bins),len(scales),len(self.Bins[1,:])))
            for j in range(len(self.Bins)):
                Coeff[j,:,:], freq = pywt.cwt(self.Bins[j,:], scales, waveletname, 1)

            if flag_training==1:       
                self.X_train = Coeff
            else:
                self.X_test = Coeff
           
    '''
    def BoXPlotAnalysis(self,flag_training):
        if flag_training==1:
            self.thr_boxplot_up = []
            self.thr_boxplot_down = []
            for i in range(9):
                x = self.X_train[:,i]
                median = np.median(x)
                upper_quartile = np.percentile(x, 75)
                lower_quartile = np.percentile(x, 25)
                iqr = upper_quartile - lower_quartile
                upper_whisker = x[x<=upper_quartile+1.5*iqr].max()
                lower_whisker = x[x>=lower_quartile-1.5*iqr].min()
                self.thr_boxplot_up.append(upper_quartile+1.5*iqr) #extreme outliers
                self.thr_boxplot_down.append(lower_quartile-1.5*iqr) #extreme outliers
        else:
             pred_y_test_boxplot=np.zeros((len(self.X_test[:,0]),9))

             pred_y_test_boxplot[:,0] = [0 if (e> self.thr_boxplot_up[0] or e< self.thr_boxplot_down[0]) else 1 for e in self.X_test[:,0]]
             pred_y_test_boxplot[:,1] = [0 if (e> self.thr_boxplot_up[1] or e< self.thr_boxplot_down[1]) else 1 for e in self.X_test[:,1]]
             pred_y_test_boxplot[:,2] = [0 if (e> self.thr_boxplot_up[2] or e< self.thr_boxplot_down[2]) else 1 for e in self.X_test[:,2]]
             pred_y_test_boxplot[:,3] = [0 if (e> self.thr_boxplot_up[3] or e< self.thr_boxplot_down[3]) else 1 for e in self.X_test[:,3]]
             pred_y_test_boxplot[:,4] = [0 if (e> self.thr_boxplot_up[4] or e< self.thr_boxplot_down[4]) else 1 for e in self.X_test[:,4]]
             pred_y_test_boxplot[:,5] = [0 if (e> self.thr_boxplot_up[5] or e< self.thr_boxplot_down[5]) else 1 for e in self.X_test[:,5]]
             pred_y_test_boxplot[:,6] = [0 if (e> self.thr_boxplot_up[6] or e< self.thr_boxplot_down[6]) else 1 for e in self.X_test[:,6]]
             pred_y_test_boxplot[:,7] = [0 if (e> self.thr_boxplot_up[7] or e< self.thr_boxplot_down[7]) else 1 for e in self.X_test[:,7]]
             pred_y_test_boxplot[:,8] = [0 if (e> self.thr_boxplot_up[8] or e< self.thr_boxplot_down[8]) else 1 for e in self.X_test[:,8]]


            # Label_testData=np.zeros(len(pred_y_test_boxplot[:,0]))
             for i in range(len(pred_y_test_boxplot[:,0])):
                 if pred_y_test_boxplot[i,0]==1 and pred_y_test_boxplot[i,1]==1 and pred_y_test_boxplot[i,2]==1 and pred_y_test_boxplot[i,3]==1 and pred_y_test_boxplot[i,4]==1 and pred_y_test_boxplot[i,5]==1 and pred_y_test_boxplot[i,6]==1 and pred_y_test_boxplot[i,7]==1 and pred_y_test_boxplot[i,8]==1:
                     Label_testData=1
                 else:
                     Label_testData=0
            
                 self.pred_test_data.append(Label_testData)
    '''
    def Standardization(self,flag_training,input_dim):
        for j in range(self.X_train.shape[0]):
            scaler = StandardScaler().fit(self.X_train[j,:,:])
            if flag_training==1:
                #self.X_train_full.append(self.X_train[j,:,:])
                self.X_train_scaled[j,:,:] = scaler.transform(self.X_train[j,:,:])
                
            else:
             #self.X_test_full.append(self.X_test[j,:,:])
                self.X_test_scaled[j,:,:] = scaler.transform(self.X_test[j,:,:])
                self.X_test_scaled = np.reshape(self.X_test_scaled, (len(self.X_test_scaled), input_dim, input_dim, 1))
        

    def evalution(self,flag_training,input_dim):
        #self.X_train_scaled= np.zeros((self.X_train.shape[0],self.X_train.shape[1],self.X_train.shape[2]))
        # self.Standardization(flag_training,input_dim)
         self.X_train_scaled=self.X_train/np.max(self.X_train)
         self.X_train_scaled = np.reshape(self.X_train_scaled, (len(self.X_train_scaled), input_dim, input_dim, 1))
        

          ############################# 
         opt=keras.optimizers.Adam()
         #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.01)#momentums = [0.0, 0.5, 0.9, 0.99]
         self.AE_CNN.compile(optimizer=opt, loss='binary_crossentropy')
         #self.AE_CNN.compile(optimizer=opt, loss="mse")
            
         history = self.AE_CNN.fit(self.X_train_scaled, self.X_train_scaled,epochs=100, batch_size=40,
                validation_split=0.1,
                shuffle=True,
                #validation_data=(X_valid, X_valid),
                #callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, mode="min")]
                )
        # self.loss_training = history.history["loss"]
         X_train_pred = self.AE_CNN.predict(self.X_train_scaled)
         self.train_mse_loss = np.mean((X_train_pred - self.X_train_scaled)**2, axis=1)
            
         threshold = np.max(self.train_mse_loss)
         median = np.median(self.train_mse_loss)
         upper_quartile = np.percentile(self.train_mse_loss, 75)
         lower_quartile = np.percentile(self.train_mse_loss, 25)
         iqr = upper_quartile - lower_quartile
         upper_whisker =self. train_mse_loss[self.train_mse_loss<=upper_quartile+1.5*iqr].max()
         lower_whisker = self.train_mse_loss[self.train_mse_loss>=lower_quartile-1.5*iqr].min()
         self.thr_boxplot.append(upper_quartile+3*iqr) #extreme outliers
         
    
    def evalution_TestData(self,flag_training):
        self.Standardization(flag_training)
        X_test_pred = self.AE_CNN.predict(self.X_test_scaled, verbose=0)
        test_mse_loss = np.mean(((X_test_pred-self.X_test_scaled))**2, axis=1)#“axis 0” represents rows and “axis 1” represents columns.
        pred_y_test = [0 if e > self.thr_boxplot[-1] else 1 for e in test_mse_loss]
        alpha_SLPF=0.05
        self.simple_LPF(pred_y_test,alpha_SLPF)
        
        self.Y_Test_pre.extend(pred_y_test)
        self.Y_Test_preLPF.extend(self.pred_y_test_simpleLPF)
        self.Time_StartCOMP.extend(self.Time_Start)
        self.Time_EndCOMP.extend(self.Time_end)
        
       
       
    def __init__(self, parent=None):

        self.X_train_full = []
        self.X_test_full = []
        self.thr_boxplot =[]
        self.Y_Test_preLPF=[]
        self.Time_StartCOMP=[]
        self.Time_EndCOMP=[]
        self.Y_Test_pre=[]
        self.pred_test_data=[]
        
        scales = range(110)
        input_dim = len(scales) #len(self.X_train)
      
       # self.model_SimpleNetwork(input_dim,encoding_dim)
        self.model_AE_CNN(input_dim)
        train = 1
        
        
        waveletname = 'morl' #'cmor1.5-1.0'
        
        start_date_train = dt.datetime(2020, 3, 1)
        start_date_train += dt.timedelta(days=0) 
        for i in range(8):
            self.x=[]
            self.X_train = []
            self.X_test = []
            
            flag_training =1
            end_date_train = start_date_train + dt.timedelta(days=10)
            end_date_train += dt.timedelta(days=0) 
            self.Find_BestDuration(flag_training,start_date_train,end_date_train,train)
            self.ConvertSignaltoIMAGE(flag_training,scales,waveletname)
            #self.X_train=np.array(self.X_train)
            self.evalution(flag_training,input_dim)
            self.BoXPlotAnalysis(flag_training)
            #########################
                ##Test
            flag_training=0
            start_date_test = end_date_train
            start_date_test += dt.timedelta(days=0) 
            end_date_test = start_date_test + dt.timedelta(days=7)
            self.Find_BestDuration(flag_training,start_date_test,end_date_test,train)
            self.ConvertSignaltoIMAGE(flag_training)
            #self.X_test=np.array(self.X_test)
            self.evalution_TestData(flag_training)
            self.BoXPlotAnalysis(flag_training)
            
            Data1=pd.DataFrame(self.Y_Test_pre)
            Data1.to_csv('Yanalog_simp_T2(9-4).csv') 
            
            Data2=pd.DataFrame(self.Time_StartCOMP)
            Data2.to_csv('TimeAnalog_simp_T2(9-4).csv') 
            
            #Data3=pd.DataFrame(self.pred_test_data)
            #Data3.to_csv('pred_test_data_Boxplot_analog_T2.csv')
            
            print("step:",i)
            start_date_train = start_date_train+dt.timedelta(days=7)
       # self.Plot_Output()
        
        '''
        self.fig, (self.ax1) = plt.subplots(1, 1)
        plt.subplots_adjust(hspace=2)
        self.line1, = self.ax1.plot([], [], 'bo', lw=2, color='k')
        self.line = [self.line1]

        self.ax1.set_xlabel('Code', fontsize=6)
        self.ax1.set_ylabel('Occurrence', fontsize=8)
        self.ax1.set_xlim(0, 7)
        self.ax1.set_ylim(0, 3700)
        self.ax1.tick_params(axis='x', labelsize=8)
        self.ax1.tick_params(axis='y', labelsize=8)
        self.ax1.grid()
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
        ani = animation.FuncAnimation(self.fig, self.test, frames=144*30, save_count=144, interval=50, repeat=False) #144 is for one day
        ani.save('C:\\Users\\davar\\Desktop\\DataSet\\apu1_12-4-21M.mp4', writer=writer)
        mng = plt.get_current_fig_manager()
       # mng.window.state('zoomed')
        mng.window.showMaximized()
        plt.show()
        '''



#
#
#


if __name__ == '__main__':
    zc = ZombieScript()

#Data=pd.DataFrame(zc.X_train)
#Data.to_csv('X_train.csv') 