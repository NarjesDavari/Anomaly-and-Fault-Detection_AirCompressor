# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:29:51 2021

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
            mycursor.execute(
                "SELECT readtimestamp, anch2, di1, di2, di3, di4, di5 ,di6, di8 FROM trainsmp where train=" +
                str(train) + " AND readtimestamp between \"" +
                startDay.strftime('%Y-%m-%d %H:%M:%S') + "\" AND \"" + endDay.strftime(
                    '%Y-%m-%d %H:%M:%S') + "\";")
            myresult = mycursor.fetchall()
            columns_name = ['timestamp', 'TP3', 'di1', 'di2', 'di3', 'di4', 'di5', 'di6', 'di8']# 'di7',
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
        
    '''  
    def model_DeepNetwor(self,input_dim,encoding_dim):
        #build model_1
        hidden_dim = int(encoding_dim / 2)
        learning_rate = 1e-5
        input_layer = tf.keras.layers.Input(shape=(input_dim, ))
        encoder = tf.keras.layers.Dense(encoding_dim, activation="relu", activity_regularizer=keras.regularizers.l1(learning_rate))(input_layer)#
        encoder = tf.keras.layers.Dense(hidden_dim, activation="relu", activity_regularizer=keras.regularizers.l1(learning_rate))(encoder)
        #encoder = tf.keras.layers.Dense(hidden_dim/2, activation="relu", activity_regularizer=keras.regularizers.l1(learning_rate))(encoder)
        encoder = tf.keras.layers.Dense(int(hidden_dim/4), activation='relu', activity_regularizer=keras.regularizers.l1(learning_rate))(encoder)#
        #decoder = tf.keras.layers.Dense(hidden_dim/2, activation="relu", activity_regularizer=keras.regularizers.l1(learning_rate))(encoder)
        decoder = tf.keras.layers.Dense(hidden_dim, activation="relu", activity_regularizer=keras.regularizers.l1(learning_rate))(encoder)
        decoder = tf.keras.layers.Dense(encoding_dim, activation="relu", activity_regularizer=keras.regularizers.l1(learning_rate))(decoder) #
        decoder = tf.keras.layers.Dense(input_dim, activation="linear")(decoder)#activation="linear", softplus
        self.autoencoder = keras.Model(inputs=input_layer, outputs=decoder)
     '''
     
     
    def model_AELSTM(self,Xtrainshape_1,Xtrainshape_2):
          # define model         
         model = tf.keras.Sequential()
         inputs = tf.keras.layers.Input(input_shape=(Xtrainshape_1, Xtrainshape_2))
         L1 = tf.keras.layers.LSTM(units=64,activation='relu',input_shape=(Xtrainshape_1, Xtrainshape_2), name='encoder_1')(inputs)
         L2 = tf.keras.layers.Dropout(rate=0.2)(L1)##we can remove this layer
         L3= tf.keras.layers.RepeatVector(n=Xtrainshape_1, name='encoder_decoder_bridge')(L2)
         L4 = tf.keras.layers.LSTM(units=64, activation='relu',return_sequences=True,name='decoder_1')(L3)
         L5 = tf.keras.layers.Dropout(rate=0.2)(L4)##we can remove this layer
         output = tf.keras.layers.TimeDistributed(keras.layers.Dense(units=Xtrainshape_2))(L5)
         self.AELSTM = keras.Model(inputs=inputs, outputs=output)
         self.AELSTM.summary()
    
    def model_Deep_AELSTM(self,Xtrainshape_1,Xtrainshape_2):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(64, kernel_initializer='he_uniform', batch_input_shape=(None, Xtrainshape_1, Xtrainshape_2), return_sequences=True, name='encoder_1'))
        model.add(tf.keras.layers.LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='encoder_2'))
        model.add(tf.keras.layers.LSTM(16, kernel_initializer='he_uniform', return_sequences=False, name='encoder_3'))
        model.add(tf.keras.layers.RepeatVector(Xtrainshape_1, name='encoder_decoder_bridge'))
        model.add(tf.keras.layers.LSTM(16, kernel_initializer='he_uniform', return_sequences=True, name='decoder_1'))
        model.add(tf.keras.layers.LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='decoder_2'))
        model.add(tf.keras.layers.LSTM(64, kernel_initializer='he_uniform', return_sequences=True, name='decoder_3'))
        model.add(tf.keras.layers.TimeDistributed(keras.layers.Dense(Xtrainshape_2)))
        model.compile(loss="mse",optimizer='adam')
        model.build()
        print(model.summary())    
    
    def simple_LPF(self,x,alpha):
        Y_LPF=np.zeros([len(x)])
        Y_LPF[0]=1
        for i in range(1,len(x)):
            #if x[i]==0:
            Y_LPF [i]=Y_LPF [i-1]+alpha*(x[i]-Y_LPF[i-1])
            #  else:
                #  Y_LPF [i]=Y_LPF [i-1]+(1-alpha)*(x[i]-Y_LPF[i-1])
        self.pred_y_test_simpleLPF=Y_LPF
    '''
    def Find_BestDuration(self,flag_training, start_date,end_date,train):   
        self.load_Data(start_date,end_date, train)
        COMP = self.data.iloc[:,2] 
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
        
        self.Time_Start =[]
        self.Time_end =[]
        self.IndexEnd=[]
        self.IndexStart=[]
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

            if (DT[i]-DT[i-1]).total_seconds()<60*5: #five min 
               if (COMP[i-1]==0 and COMP[i]==1): #low to high
                  t_LH.append(DT[i])
                  flag_LH =1
                  index_LH.append(i)
            if (DT[i]-DT[i-1]).total_seconds()<60*5:  #five min  
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
            
            if (DT[i]-DT[i-1]).total_seconds()>60*5 and flag_HL ==0 and flag_LH==0:  #case a
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
            if flag_LH_after_b==1 and flag_interrupt_a ==1 and flag_HL==0 and flag_LH==0:
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
                    if t_HL[-1]>t_LH[-1] and (t_HL[-1]-t_LH[-1]).total_seconds()<200:
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

              
                   self.Time_Start.append(t_HL[-2])
                   self.Time_end.append(t_HL[-1])
                   self.IndexEnd.append(index_HL[-1])
                   self.IndexStart.append(index_HL[-2])
                   flag_LH=0
                   flag_HL=0            
              


    '''
    
    def Find_BestDuration(self,flag_training, start_date,end_date,train):   
        self.load_Data(start_date,end_date, train)
        COMP = self.data.iloc[:,2] 
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
        
        self.Time_Start =[]
        self.Time_end =[]
        self.IndexEnd=[]
        self.IndexStart=[]
        self.dur_high_comp=[]
        self.dur_low_comp=[]
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

              
                   self.Time_Start.append(t_HL[-2])
                   self.Time_end.append(t_HL[-1])
                   self.dur_low_comp.append((t_LH[-1]-t_HL[-2]).total_seconds())
                   self.dur_high_comp.append((t_HL[-1]-t_LH[-1]).total_seconds())
                   self.IndexEnd.append(index_HL[-1])
                   self.IndexStart.append(index_HL[-2])
                   flag_LH=0
                   flag_HL=0 
    
    def DigitalSensor_analysis(self,flag_training):

      #  self.ax1.set_title('Binary Code ' + str(single_date), fontsize=8)
        if len(self.IndexStart)>0:
            mydict = {

            }           
            for index in range(len(self.IndexStart)): 
                #  aux = ''
                mydict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}#
                for j in range(self.IndexStart[index],self.IndexEnd[index]):
                    aux = ''
                    c=0
                    for valu in self.data.iloc[j, 2:9].values:
                        if valu<1:
                            #if c not in mydict.keys(): 
                                #   mydict[c] = 1
                                #  else:
                            mydict[c] += 1
                        c+=1
           
                   # def keyfunction(k):
                      #  return mydict[k]

                self.x = []
                self.y = []
                for key in range(7): #sorted(mydict, key=keyfunction, reverse=True)[:10]:
                    self.x.append(int(key))
                    self.y.append(mydict[key])
                 #   print("key: "+str(key)+" value: "+str(mydict[key]))


                self.y.append(self.dur_low_comp[index])
                self.y.append(self.dur_high_comp[index])
                if flag_training==1:       
                    self.X_train.append(self.y)
                else:
                    self.X_test.append(self.y)
           
    
    def Standardization(self,flag_training):
         scaler = StandardScaler().fit(self.X_train)
         if flag_training==1:
             self.X_train_full.append(self.X_train)
             self.X_train_scaled = scaler.transform(self.X_train)
         else:
             self.X_test_full.append(self.X_test)
             self.X_test_scaled = scaler.transform(self.X_test)
   
    def create_dataset(self,flag_training,Data,len_subsequences,N_subsequences):
        samples=list()
        for i in range(0,N_subsequences,len_subsequences):
            sample = Data[i:i+len_subsequences,:]
            samples.append(sample)
        print(len(samples))
        data = np.array(samples) # convert list of arrays into 2d array
        print(data.shape)
        Data_scaled = data.reshape((len(samples), len_subsequences, Data.shape[1]))# reshape into [samples, timesteps, features]
        return Data_scaled
        
    def evalution(self,flag_training):
        len_subseq = 200 
        N_subseq = int(self.X_train.shape[0]/len_subseq)
        self.Standardization(flag_training)
        Data =self.X_train_scaled
        self.X_train_scaled = self.create_dataset(flag_training,Data,len_subseq,N_subseq) 

          ############################# 
        Xtrainshape_2 = 9
        Xtrainshape_1 = self.X_train_scaled.shape[1]
        self.model_AELSTM(Xtrainshape_1,Xtrainshape_2)
        opt=keras.optimizers.Adam()
         #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.01)#momentums = [0.0, 0.5, 0.9, 0.99]
        self.AELSTM.compile(optimizer=opt, loss="mse")
            
        history = self.AELSTM.fit(self.X_train_scaled, self.X_train_scaled,epochs=100, batch_size=40,
                validation_split=0.1,
                #shuffle=False,
                #shuffle=True,
                #validation_data=(X_valid, X_valid),
                callbacks=[
                         keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, mode="min")])
        # self.loss_training = history.history["loss"]
        X_train_pred = self.LSTM.predict(self.X_train_scaled)
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
        len_subseq = self.X_test.shape[0]
        N_subseq = 1
        self.Standardization(flag_training)
        Data =self.X_test_scaled
        self.X_test_scaled= self.create_dataset(flag_training,Data,len_subseq,N_subseq) 

        X_test_pred = self.LSTM.predict(self.X_test_scaled, verbose=0)
        test_mse_loss = np.mean(((X_test_pred-self.X_test_scaled))**2, axis=1)#“axis 0” represents rows and “axis 1” represents columns.
        pred_y_test = [0 if e > self.thr_boxplot[-1] else 1 for e in test_mse_loss]
        alpha_SLPF=0.05
        self.simple_LPF(pred_y_test,alpha_SLPF)
        
        self.Y_Test_pre.extend(pred_y_test)
        self.Y_Test_preLPF.extend(self.pred_y_test_simpleLPF)
        self.Time_StartCOMP.extend(self.Time_Start)
        self.Time_EndCOMP.extend(self.Time_end)
  #  def Plot_Output(self):
        x=np.zeros(len(self.pred_y_test_simpleLPF))
        x[0:len(self.pred_y_test_simpleLPF)]=0.5
       # plt.figure(1)
        plt.plot(self.Time_Start,self.pred_y_test_simpleLPF,'-p', color='gray',
                 markersize=3, linewidth=2,
                 markerfacecolor='blue',
                 markeredgecolor='blue',
                 markeredgewidth=2)
        plt.plot(self.Time_Start,x,'red')
        plt.ylim([0, 1])
        plt.xlabel("Date/Time")
        plt.ylabel("The filtered Label of test data")
        plt.legend(["SimpleLPF_boxplot","SimpleLPF_boxplot_mae","SimpleLPF_boxplot_mre","ButterWorth"],loc ="upper right")#"before filtering",
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.show()
        '''
        i=0
        while i<= (len(self.pred_y_test_simpleLPF)):
            flag_end_duration=0
            k=0
            while flag_end_duration==0:
                if  i<len(self.pred_y_test_simpleLPF):
                    if self.pred_y_test_simpleLPF[i]<0.5: 
                        if k==0:
                            print("Start Time of Alarm",self.Time_Start[i])
                            k=1
                    else:
                        flag_end_duration=1
                i=i+1
                if i>= (len(self.pred_y_test_simpleLPF)):
                    break
            if flag_end_duration==1 and k==1:
                print("End Time of Alarm",self.Time_Start[i])   
    
         '''
       
    def __init__(self, parent=None):

        self.X_train_full = []
        self.X_test_full = []
        self.thr_boxplot =[]
        self.Y_Test_preLPF=[]
        self.Time_StartCOMP=[]
        self.Time_EndCOMP=[]
        self.Y_Test_pre=[]
        
        input_dim = 9 #len(self.X_train)
        encoding_dim = 7#int(input_dim/2)
        #self.model_SimpleNetwork(input_dim,encoding_dim)
        
        train = 1
        
        start_date_train = dt.datetime(2020, 6, 1)
        start_date_train += dt.timedelta(days=0) 
        for i in range(17):
            self.x=[]
            self.X_train = []
            self.X_test = []

            flag_training =1
            end_date_train = start_date_train + dt.timedelta(days=2)
            end_date_train += dt.timedelta(days=0) 
            self.Find_BestDuration(flag_training,start_date_train,end_date_train,train)
            self.DigitalSensor_analysis(flag_training)
            self.X_train=np.array(self.X_train)
            self.evalution(flag_training)
            #########################
                ##Test
            flag_training=0
            start_date_test = end_date_train
            end_date_test = start_date_test + dt.timedelta(days=1)
            end_date_test += dt.timedelta(days=0)
            self.Find_BestDuration(flag_training,start_date_test,end_date_test,train)
            self.DigitalSensor_analysis(flag_training)
            self.X_test=np.array(self.X_test)
            self.evalution_TestData(flag_training)
            
            Data=pd.DataFrame(self.Y_Test_pre)
            Data.to_csv('Y_Test_predict_Simple_9inp_zero.csv') 
            
            Data2=pd.DataFrame(self.Time_StartCOMP)
            Data2.to_csv('Time_StartCOMP_Simple_9inp_zero.csv') 
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