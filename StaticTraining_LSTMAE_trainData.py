# -*- coding: utf-8 -*-
"""
Created on Sun May 16 22:45:44 2021

@author: davari
"""

import datetime as dt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

#matplotlib.use('Qt5Agg')  
#matplotlib.use('TkAgg')
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler,MaxAbsScaler,Normalizer
from sklearn.model_selection import train_test_split
import datetime
tf.compat.v1.random.set_random_seed(1)
from keras.models import Sequential
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
from time import time
#%load_ext tensorboard
from IPython.display import display, HTML

class ZombieScript:
    '''
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
            '''
    def load_Data_csvfile(self):
        #DataSet = pd.read_csv('C:/Users/davar/Documents/INESCTEC/code_python/Autoencoder/newData_21March2022/LargeDataForTraining_time.csv') #
        #DataSet_testAir = pd.read_csv('C:/Users/davar/Documents/INESCTEC/code_python/Autoencoder/newData_21March2022/MetroAir2_time.csv')
        #DataSet_testOil = pd.read_csv('C:/Users/davar/Documents/INESCTEC/code_python/Autoencoder/newData_21March2022/MetroOil2_time.csv')
        DataSet = pd.read_csv('C:/Users/davar/Documents/INESCTEC/code_python/Autoencoder/sendToShazia/MetroPT_1.csv')
        #DataSet_Testfull = pd.concat ((DataSet_testAir,DataSet_testOil),ignore_index=True)
        DataSet_trainig = DataSet[1:2000000]
        DataSet_Testfull = DataSet[2000000:]
        columns_name = ['timestamp', 'TP2',	'TP3',	'H1', 'DV_pressure', 'Reservoirs',	'Oil_temperature',	'Flowmeter',	'Motor_current','COMP', 'DV_eletric',	'Towers',	'MPG',	'LPS',	'Pressure_switch',	'Oil_level',	'Caudal_impulses']#timestamp	TP2	TP3	H1	DV_pressure	Reservoirs	Oil_temperature	Flowmeter	Motor_current	COMP	DV_eletric	Towers	MPG	LPS	Pressure_switch	Oil_level	Caudal_impulses

        DataSet_Train = DataSet_trainig[columns_name]
        DataSet_Test = DataSet_Testfull[columns_name]
        # if there is missing value: DataSet_Testfull.dropna(inplace=True, axis=1)
        return DataSet_Train,DataSet_Test
    
    def model_SimpleNetwork(self,input_dim,encoding_dim):
        ##build model_0
        #input_dim = self.X_train.shape[1] 
        input_layer = tf.keras.layers.Input(shape=(input_dim, )) 
        encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer) #,activity_regularizer=keras.regularizers.l1(10e-5)
        decoded = tf.keras.layers.Dense(input_dim, activation='linear')(encoded)#linear, softplus,tanh,  sigmoid( Sigmoid is a binary classifier, should not use it)
        self.autoencoder = keras.Model(input_layer, decoded) # map an input to its reconstruction
        self.autoencoder.summary()
      
    def model_DeepNetwor(self,input_dim,encoding_dim):
        #build model_1
        hidden_dim = int(encoding_dim / 2)
        learning_rate = 1e-3
        input_layer = tf.keras.layers.Input(shape=(input_dim, ))
        encoder = tf.keras.layers.Dense(encoding_dim, activation="relu", activity_regularizer=keras.regularizers.l1(learning_rate))(input_layer)#
        encoder = tf.keras.layers.Dense(hidden_dim, activation="relu", activity_regularizer=keras.regularizers.l1(learning_rate))(encoder)
        encoder = tf.keras.layers.Dense(int(hidden_dim/2), activation="relu", activity_regularizer=keras.regularizers.l1(learning_rate))(encoder)
        encoder = tf.keras.layers.Dense(int(hidden_dim/5), activation='relu', activity_regularizer=keras.regularizers.l1(learning_rate))(encoder)#
        decoder = tf.keras.layers.Dense(int(hidden_dim/2), activation="relu", activity_regularizer=keras.regularizers.l1(learning_rate))(encoder)
        decoder = tf.keras.layers.Dense(hidden_dim, activation="relu", activity_regularizer=keras.regularizers.l1(learning_rate))(encoder)
        decoder = tf.keras.layers.Dense(encoding_dim, activation="relu", activity_regularizer=keras.regularizers.l1(learning_rate))(decoder) #
        decoder = tf.keras.layers.Dense(input_dim, activation="sigmoid")(decoder)#activation="linear", softplus
        self.autoencoder = keras.Model(inputs=input_layer, outputs=decoder)
        self.autoencoder.summary()

    def model_Deep_AELSTM(self,n_features,encoding_dim,n_steps):
        model = Sequential()
        model.add(tf.keras.layers.LSTM(encoding_dim, input_shape=(n_steps,n_features), return_sequences=True)) #, activation='relu', n_steps=X_train.shape[1], n_features=X_train.shape[2]
        model.add(tf.keras.layers.LSTM(int(encoding_dim/2), activation="relu", return_sequences=False))
        model.add(tf.keras.layers.Dropout(rate=0.2))
        model.add(tf.keras.layers.RepeatVector(n=n_steps))
        # define reconstruct decoder
        model.add(tf.keras.layers.LSTM(int(encoding_dim/2), activation="relu", return_sequences=True))
        model.add(tf.keras.layers.LSTM(encoding_dim, activation="relu",  return_sequences=True))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features, activation="sigmoid")))
        self.AELSTM=model
        self.AELSTM.summary()
   
    def model_Deep_AELSTM_2(self,n_features,encoding_dim,n_steps):
        model = Sequential()
        model.add(tf.keras.layers.LSTM(encoding_dim, input_shape=(n_steps,n_features), return_sequences=True)) #, activation='relu', n_steps=X_train.shape[1], n_features=X_train.shape[2]
        model.add(tf.keras.layers.LSTM(int(encoding_dim), activation="relu", return_sequences=True))
        model.add(tf.keras.layers.LSTM(int(encoding_dim/2), activation="relu", return_sequences=True))
        model.add(tf.keras.layers.Dropout(rate=0.2))
        model.add(tf.keras.layers.LSTM(int(encoding_dim/4), activation="relu", return_sequences=False))
        model.add(tf.keras.layers.Dropout(rate=0.2))
        model.add(tf.keras.layers.RepeatVector(n=n_steps))
        # define reconstruct decoder
        model.add(tf.keras.layers.LSTM(int(encoding_dim/4), activation="relu", return_sequences=True))
        model.add(tf.keras.layers.Dropout(rate=0.2))
        model.add(tf.keras.layers.LSTM(int(encoding_dim/2), activation="relu", return_sequences=True))
        model.add(tf.keras.layers.LSTM(int(encoding_dim), activation="relu", return_sequences=True))
        model.add(tf.keras.layers.LSTM(encoding_dim, activation="relu",  return_sequences=True))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features, activation="sigmoid")))
        self.AELSTM=model
        self.AELSTM.summary()
        
    def model_Deep_AELSTM_3(self,n_features,encoding_dim,n_steps):
        model = Sequential()
        model.add(tf.keras.layers.LSTM(encoding_dim, input_shape=(n_steps,n_features), return_sequences=True)) #, activation='relu', n_steps=X_train.shape[1], n_features=X_train.shape[2]
        model.add(tf.keras.layers.LSTM(int(encoding_dim/2), activation="relu", return_sequences=False))
        model.add(tf.keras.layers.Dropout(rate=0.2))
        model.add(tf.keras.layers.LSTM(int(encoding_dim/4), activation="relu", return_sequences=True))
        model.add(tf.keras.layers.RepeatVector(n=n_steps))
        # define reconstruct decoder
        model.add(tf.keras.layers.LSTM(int(encoding_dim/4), activation="relu", return_sequences=True))
        model.add(tf.keras.layers.Dropout(rate=0.2))
        model.add(tf.keras.layers.LSTM(int(encoding_dim/2), activation="relu", return_sequences=True))
        model.add(tf.keras.layers.LSTM(encoding_dim, activation="relu",  return_sequences=True))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features, activation="sigmoid")))
        self.AELSTM=model
        self.AELSTM.summary() 
        
    
    def simple_LPF(self,x,alpha):
        Y_LPF=np.zeros([len(x)])
        Y_LPF[0]=1
        for i in range(1,len(x)):
            #if x[i]==0:
            Y_LPF [i]=Y_LPF [i-1]+alpha*(x[i]-Y_LPF[i-1])
            #  else:
                #  Y_LPF [i]=Y_LPF [i-1]+(1-alpha)*(x[i]-Y_LPF[i-1])
        self.pred_y_test_simpleLPF=Y_LPF


    def Find_BestDuration1(self,data,flag_training):   
        COMP = data['COMP'] 
        Motor_current = data['Motor_current'] 
        TP3 = data['TP3'] 
        TP2 = data['TP2']
        H1 = data['H1']
        Flowmeter = data['Flowmeter']
        Oil_temp =data['Oil_temperature']
        DV_Pre =data['DV_pressure']
        Reservoirs =data['Reservoirs']
        #TP3 = 16 * (1 - ((20 - TP3) / 16))
        #timestampSecond
        DT = data['timestamp'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())        
        #data['timestamp'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())  #.%f
        flag_LH =0
        flag_HL =0
        flag_interrupt_b =0
        flag_interrupt_a =0
    
        t_HL=[]
        t_LH=[]
   
        index_HL=[]
        index_LH=[]
        k=2
        flag_start=0
        flag_LH_after_b=0
        flag_LH_after_a=0
        flag_msg=0
        num_bin_inc=2
        num_bin_dec=5
        self.Time_Start =[]
        self.Time_end =[]
        self.IndexEnd=[]
        self.IndexStart=[]
        self.dur_high_comp=[]
        self.dur_low_comp=[]
        self.Bins=[] #np.empty((0,5*7+3+2), float)
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

            if flag_training==1 and int(np.ceil(DT[i]-DT[i-1]))<60*5 or flag_training==0: #five min 
               if (COMP[i-1]==0 and COMP[i]==1): #low to high
                  t_LH.append(DT[i])
                  flag_LH =1
                  index_LH.append(i) 
               if (COMP[i-1]==1 and COMP[i]==0):#high to low
                  t_HL.append(DT[i])
                  flag_HL = 1
                  index_HL.append(i)
   
            if flag_HL ==1 and flag_LH ==1 and int(np.ceil(abs(t_LH[-1]-t_HL[-1])))<60:
                flag_HL =0
                flag_LH =0
                del t_LH[-1]
                del index_LH[-1]
                del t_HL[-1]
                del index_HL[-1]
            
            if int(np.ceil(DT[i]-DT[i-1]))>60*5 and flag_HL ==0 and flag_LH==0:  #and flag_training==1 case a
                flag_interrupt_a =1
     
            if int(np.ceil(DT[i]-DT[i-1]))>60*5 and flag_LH ==1 and flag_HL ==0:  # and flag_training==1 case b
                flag_LH=0
                del t_LH[-1]
                del index_LH[-1]
                flag_interrupt_b =1
       
            if flag_interrupt_b ==1 and flag_LH ==1 and flag_HL==0: # and flag_training==1:
                flag_LH =0
                flag_interrupt_b =0
                del t_LH[-1]
                del index_LH[-1]
                flag_LH_after_b =1

      
            if flag_interrupt_b ==1 and flag_LH==0 and flag_HL ==1: # and flag_training==1:
                flag_HL =0
                flag_interrupt_b =0
    
            if flag_interrupt_a ==1 and flag_LH==1 and flag_HL==0: #and flag_training==1:
                flag_LH=0
                flag_interrupt_a =0
                del t_LH[-1]
                del index_LH[-1]
                flag_LH_after_a =1
            if flag_LH_after_a ==1 and flag_HL==1: #and flag_training==1:
                flag_LH_after_a=0
                flag_HL=0
            if flag_LH_after_b==1 and flag_interrupt_a ==1 and flag_HL==0 and flag_LH==0: #and flag_training==1:
                flag_LH_after_b=0
        
            if flag_interrupt_a ==1 and flag_HL==1 and flag_LH==0: # and flag_training==1:
                if flag_interrupt_b==1:
                    flag_interrupt_b=0
                flag_HL=0
                flag_interrupt_a =0
        
            if flag_LH_after_b==1 and flag_HL==1 and flag_interrupt_a==0: #and flag_training==1:
                flag_LH_after_b=0 
                flag_HL=0 
       
            if flag_HL==1 and flag_LH==1: 
                if flag_training==1:
                   if  int(np.ceil(t_HL[-1]-t_LH[-1]))<1000 or int(np.ceil(t_HL[-1]-t_LH[-1]))>3000: ## t_HL[-1]>t_LH[-1] and#and 60<(t_HL[len(t_HL)-1]-t_LH[len(t_LH)-1]): 
                       flag_HL=0
                       flag_LH=0
                       del t_LH[-1]
                       del index_LH[-1] 
                else:
                    if t_HL[-1]>t_LH[-1] and int(np.ceil(t_HL[-1]-t_LH[-1]))<120:
                        flag_HL=0
                        flag_LH=0
                        del t_LH[-1]
                        del index_LH[-1]
                        del t_HL[-1]
                        del index_HL[-1]

            if flag_msg==0 and flag_LH==0 and flag_HL==0 and flag_training==0 and COMP[i]==0 and flag_interrupt_a==0 and flag_interrupt_b==0 and flag_LH_after_b==0 and flag_LH_after_a==0: 
               count_w_comp = count_w_comp+1
               if count_w_comp>500 and int(np.ceil(DT[i]-t_HL[-1]))>420:
                   print("Comp is Working more than",int(np.ceil(DT[i]-t_HL[-1])))
                   flag_msg = 1
                   T_start_msg.append(DT[i])
            if flag_msg==1 and flag_LH==1 and flag_HL==0 and flag_interrupt_a==0:
                print("Comp Worked for more than", int(np.ceil(t_LH[-1]-t_HL[-1])))
                count_w_comp=0
                flag_msg=0
                work_time_more420.append(int(np.ceil(t_LH[-1]-t_HL[-1])))
                T_end_msg.append(DT[i])
            elif flag_msg==1 and flag_LH==0 and flag_HL==0 and flag_interrupt_a==1:
                print("Comp Worked for more than", int(np.ceil(DT[i]-t_HL[-1])))
                count_w_comp=0
                flag_msg=0
                work_time_more420.append(int(np.ceil(DT[i]-t_HL[-1])))
                T_end_msg.append(DT[i])
           
            if flag_HL==1 and flag_LH==1: 

                if int(np.ceil(t_LH[-1]-t_HL[-2]))<60:
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
                elif flag_training==1 and t_LH[-1]> t_HL[-2] and int(np.ceil(t_LH[-1]-t_HL[-2]))>420:
                   flag_HL=0
                   flag_LH=0
                   del t_LH[-1]
                   del index_LH[-1]     
                else:
                   #  data = [TP2,TP3,H1, Flowmeter, Motor_current]
                   # for i in range(len(data)+1):
                      #   B1,B2,B3,B4,B5,B6,B7= self.Extract_Bins(data[i:],index_LH,index_HL,t_HL,num_bin_inc,num_bin_dec)
                    
                     index11 = int((index_LH[-1]-index_HL[-2])/num_bin_inc + index_HL[-2])
                     B1_tp2=np.mean(TP2[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])
                     B1_tp3=np.mean(TP3[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])
                     B1_H1=np.mean(H1[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])
                     B1_FM=np.mean(Flowmeter[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])
                     B1_MC=np.mean(Motor_current[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])#.total_seconds()
                     
                     B2_tp2=np.mean(TP2[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])
                     B2_tp3=np.mean(TP3[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])#.total_seconds()
                     B2_H1=np.mean(H1[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])
                     B2_FM=np.mean(Flowmeter[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])
                     B2_MC=np.mean(Motor_current[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])#.total_seconds()

                     index21 = int((index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
                     B3_tp2=np.mean(TP2[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])
                     B3_tp3=np.mean(TP3[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])#.total_seconds()
                     B3_H1=np.mean(H1[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])
                     B3_FM=np.mean(Flowmeter[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])
                     B3_MC=np.mean(Motor_current[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])#.total_seconds()
              
                
                     index22 = int(2*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
                     B4_tp2=np.mean(TP2[index21:index22])*(t_HL[-1]-t_HL[-2])
                     B4_tp3=np.mean(TP3[index21:index22])*(t_HL[-1]-t_HL[-2])#.total_seconds()
                     B4_H1=np.mean(H1[index21:index22])*(t_HL[-1]-t_HL[-2])
                     B4_FM=np.mean(Flowmeter[index21:index22])*(t_HL[-1]-t_HL[-2])
                     B4_MC=np.mean(Motor_current[index21:index22])*(t_HL[-1]-t_HL[-2])#.total_seconds()
               
                
                     index23 = int(3*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
                     B5_tp2=np.mean(TP2[index22:index23])*(t_HL[-1]-t_HL[-2])
                     B5_tp3=np.mean(TP3[index22:index23])*(t_HL[-1]-t_HL[-2])#.total_seconds()
                     B5_H1=np.mean(H1[index22:index23])*(t_HL[-1]-t_HL[-2])
                     B5_FM=np.mean(Flowmeter[index22:index23])*(t_HL[-1]-t_HL[-2])
                     B5_MC=np.mean(Motor_current[index22:index23])*(t_HL[-1]-t_HL[-2])#.total_seconds()
              

                     index24 = int(4*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
                     B6_tp2=np.mean(TP2[index23:index24])*(t_HL[-1]-t_HL[-2])
                     B6_tp3=np.mean(TP3[index23:index24])*(t_HL[-1]-t_HL[-2])#.total_seconds()
                     B6_H1=np.mean(H1[index23:index24])*(t_HL[-1]-t_HL[-2])
                     B6_FM=np.mean(Flowmeter[index23:index24])*(t_HL[-1]-t_HL[-2])
                     B6_MC=np.mean(Motor_current[index23:index24])*(t_HL[-1]-t_HL[-2])#.total_seconds()
              

                     index25 = int(5*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
                     B7_tp2=np.mean(TP2[index24:index25])*(t_HL[-1]-t_HL[-2])
                     B7_tp3=np.mean(TP3[index24:index25])*(t_HL[-1]-t_HL[-2])
                     B7_H1=np.mean(H1[index24:index25])*(t_HL[-1]-t_HL[-2])#.total_seconds()
                     B7_FM=np.mean(Flowmeter[index24:index25])*(t_HL[-1]-t_HL[-2])
                     B7_MC=np.mean(Motor_current[index24:index25])*(t_HL[-1]-t_HL[-2])#.total_seconds()
              
                     self.Time_Start.append(datetime.fromtimestamp(t_HL[-2]))
                     self.Time_end.append(datetime.fromtimestamp(t_HL[-1]))
                     self.dur_low_comp.append(int(np.ceil(t_LH[-1]-t_HL[-2])))#.total_seconds())
                     self.dur_high_comp.append(int(np.ceil(t_HL[-1]-t_LH[-1])))#.total_seconds())
                     self.IndexEnd.append(index_HL[-1])
                     self.IndexStart.append(index_HL[-2])
                     
                     TL=(t_LH[-1]-t_HL[-2])#.total_seconds()
                     TH=(t_HL[-1]-t_LH[-1])#.total_seconds()
                     BIN=[]
                     #bbin=[]
                     Bin_tp2 =[B1_tp2,B2_tp2,B3_tp2,B4_tp2,B5_tp2,B6_tp2,B7_tp2]
                     BIN.extend(Bin_tp2)
                    # bbin.extend(Bin_tp2)
                     Bin_tp3 =[B1_tp3,B2_tp3,B3_tp3,B4_tp3,B5_tp3,B6_tp3,B7_tp3]
                     BIN.extend(Bin_tp3)
                    # bbin.extend(Bin_tp3)
                     Bin_H1 =[B1_H1,B2_H1,B3_H1,B4_H1,B5_H1,B6_H1,B7_H1]
                     BIN.extend(Bin_H1)
                     Bin_FM =[B1_FM,B2_FM,B3_FM,B4_FM,B5_FM,B6_FM,B7_FM]
                     BIN.extend(Bin_FM)
                     Bin_MC =[B1_MC,B2_MC,B3_MC,B4_MC,B5_MC,B6_MC,B7_MC]
                     BIN.extend(Bin_MC)
                     
                     Min_Oil = np.min(Oil_temp[index11:index25])
                     Max_Oil = np.max(Oil_temp[index11:index25])
                     MA1_Oil = np.mean(Oil_temp[index_HL[-2]:index_LH[-1]])
                     MA2_Oil = np.mean(Oil_temp[index_LH[-1]:index_HL[-1]])
                     Med_DV = np.median(DV_Pre[index11:index25])
                     Med_Reser = np.median(Reservoirs[index11:index25])
                     ss=[Min_Oil,Max_Oil,MA1_Oil,MA2_Oil,Med_DV,Med_Reser,TL,TH]
                     ss1=[TL,TH]
                     BIN.extend(ss)
                     
                    # bbin.extend(ss1)
                    # self.Bins.append(bbin)
                     self.Bins.append(BIN)
                     flag_LH=0
                     flag_HL=0          


    def Feature_values(self,TP2,TP3,H1,Motor_current,Oil_temp,DV_Pre,Reservoirs,num_bin_inc,num_bin_dec,t_LH,t_HL,index_LH,index_HL):
        index11 = int((index_LH[-1]-index_HL[-2])/num_bin_inc + index_HL[-2])
        B1_tp2=np.mean(TP2[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])
        B1_tp3=np.mean(TP3[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])
        B1_H1=np.mean(H1[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])
       # B1_FM=np.mean(Flowmeter[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])
        B1_MC=np.mean(Motor_current[index_HL[len(index_HL)-1-1]:index11])*(t_HL[-1]-t_HL[-2])#.total_seconds()
                     
        B2_tp2=np.mean(TP2[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])
        B2_tp3=np.mean(TP3[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])#.total_seconds()
        B2_H1=np.mean(H1[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])
        #B2_FM=np.mean(Flowmeter[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])
        B2_MC=np.mean(Motor_current[index11:index_LH[-1]])*(t_HL[-1]-t_HL[-2])#.total_seconds()

        index21 = int((index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
        B3_tp2=np.mean(TP2[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])
        B3_tp3=np.mean(TP3[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])#.total_seconds()
        B3_H1=np.mean(H1[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])
       # B3_FM=np.mean(Flowmeter[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])
        B3_MC=np.mean(Motor_current[index_LH[-1]:index21])*(t_HL[-1]-t_HL[-2])#.total_seconds()
                 
        index22 = int(2*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
        B4_tp2=np.mean(TP2[index21:index22])*(t_HL[-1]-t_HL[-2])
        B4_tp3=np.mean(TP3[index21:index22])*(t_HL[-1]-t_HL[-2])#.total_seconds()
        B4_H1=np.mean(H1[index21:index22])*(t_HL[-1]-t_HL[-2])
       # B4_FM=np.mean(Flowmeter[index21:index22])*(t_HL[-1]-t_HL[-2])
        B4_MC=np.mean(Motor_current[index21:index22])*(t_HL[-1]-t_HL[-2])#.total_seconds()
               
                
        index23 = int(3*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
        B5_tp2=np.mean(TP2[index22:index23])*(t_HL[-1]-t_HL[-2])
        B5_tp3=np.mean(TP3[index22:index23])*(t_HL[-1]-t_HL[-2])#.total_seconds()
        B5_H1=np.mean(H1[index22:index23])*(t_HL[-1]-t_HL[-2])
       # B5_FM=np.mean(Flowmeter[index22:index23])*(t_HL[-1]-t_HL[-2])
        B5_MC=np.mean(Motor_current[index22:index23])*(t_HL[-1]-t_HL[-2])#.total_seconds()
              
        index24 = int(4*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
        B6_tp2=np.mean(TP2[index23:index24])*(t_HL[-1]-t_HL[-2])
        B6_tp3=np.mean(TP3[index23:index24])*(t_HL[-1]-t_HL[-2])#.total_seconds()
        B6_H1=np.mean(H1[index23:index24])*(t_HL[-1]-t_HL[-2])
        #B6_FM=np.mean(Flowmeter[index23:index24])*(t_HL[-1]-t_HL[-2])
        B6_MC=np.mean(Motor_current[index23:index24])*(t_HL[-1]-t_HL[-2])#.total_seconds()
              
        index25 = int(5*(index_HL[-1]-index_LH[-1])/num_bin_dec + index_LH[-1])
        B7_tp2=np.mean(TP2[index24:index25])*(t_HL[-1]-t_HL[-2])
        B7_tp3=np.mean(TP3[index24:index25])*(t_HL[-1]-t_HL[-2])
        B7_H1=np.mean(H1[index24:index25])*(t_HL[-1]-t_HL[-2])#.total_seconds()
       # B7_FM=np.mean(Flowmeter[index24:index25])*(t_HL[-1]-t_HL[-2])
        B7_MC=np.mean(Motor_current[index24:index25])*(t_HL[-1]-t_HL[-2])#.total_seconds()
              
        Time_Start=(datetime.fromtimestamp(t_HL[-2]))
        Time_end=(datetime.fromtimestamp(t_HL[-1]))
        dur_low_comp=(int(np.ceil(t_LH[-1]-t_HL[-2])))#.total_seconds())
        dur_high_comp=(int(np.ceil(t_HL[-1]-t_LH[-1])))#.total_seconds())
                     
        TL=(t_LH[-1]-t_HL[-2])#.total_seconds()
        TH=(t_HL[-1]-t_LH[-1])#.total_seconds()
        BIN=[]
                     #bbin=[]
        Bin_tp2 =[B1_tp2,B2_tp2,B3_tp2,B4_tp2,B5_tp2,B6_tp2,B7_tp2]
        BIN.extend(Bin_tp2)
                    # bbin.extend(Bin_tp2)
        Bin_tp3 =[B1_tp3,B2_tp3,B3_tp3,B4_tp3,B5_tp3,B6_tp3,B7_tp3]
        BIN.extend(Bin_tp3)
                    # bbin.extend(Bin_tp3)
        Bin_H1 =[B1_H1,B2_H1,B3_H1,B4_H1,B5_H1,B6_H1,B7_H1]
        BIN.extend(Bin_H1)
       # Bin_FM =[B1_FM,B2_FM,B3_FM,B4_FM,B5_FM,B6_FM,B7_FM]
       #BIN.extend(Bin_FM)
        Bin_MC =[B1_MC,B2_MC,B3_MC,B4_MC,B5_MC,B6_MC,B7_MC]
        BIN.extend(Bin_MC)
                     
        Min_Oil = np.min(Oil_temp[index11:index25])
        Max_Oil = np.max(Oil_temp[index11:index25])
        MA1_Oil = np.mean(Oil_temp[index_HL[-2]:index_LH[-1]])
        MA2_Oil = np.mean(Oil_temp[index_LH[-1]:index_HL[-1]])
        Med_DV = np.median(DV_Pre[index11:index25])
        Med_Reser = np.median(Reservoirs[index11:index25])
        ss=[Min_Oil,Max_Oil,MA1_Oil,MA2_Oil,Med_DV,Med_Reser,TL,TH]
        BIN.extend(ss)
        return BIN,Time_Start,Time_end,dur_low_comp,dur_high_comp
    
    

    def Find_BestDuration(self,data,flag_training):   
        COMP = data['COMP'] 
        Motor_current = data['Motor_current'] 
        TP3 = data['TP3'] 
        TP2 = data['TP2']
        H1 = data['H1']
       # Flowmeter = data['Flowmeter']
        Oil_temp =data['Oil_temperature']
        DV_Pre =data['DV_pressure']
        Reservoirs =data['Reservoirs']
        #TP3 = 16 * (1 - ((20 - TP3) / 16))
        #timestampSecond
        #DT = data['timestamp'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())        
        DT = data['timestamp'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())  #
        flag_LH =0
        flag_HL =0
        flag_interrupt_b =0
        flag_interrupt_a =0
    
        t_HL=[]
        t_LH=[]
   
        index_HL=[]
        index_LH=[]
        k=2
        flag_start=0
        flag_LH_after_b=0
        flag_LH_after_a=0
        flag_msg=0
        flag_msg_w1=0
        num_bin_inc=2
        num_bin_dec=5
        Time_Start =[]
        Time_end =[]
        IndexEnd=[]
        IndexStart=[]
        dur_high_comp=[]
        dur_low_comp=[]
        Bins=[] #np.empty((0,5*7+3+2), float)
        T_start_msg=[]
        T_end_msg=[]
        work_time_more420=[]
        count_w_comp=0
        count_w_comp1=0
        while flag_start==0:
            if COMP[k-1]==1 and COMP[k]==0 and TP3[k]>8:
                t_HL_0 =DT[k]
                t_HL.append(t_HL_0)
                index_HL.append(k)
                flag_start=1
            else:
                k=k+1
        for i in range(k+1,len(COMP)):#
        
            if COMP[i]==0 and COMP[i-1]==1 and COMP[i+1]==1:
               COMP[i] =1         

            if int(np.ceil(DT[i]-DT[i-1]))<60*5: #five min 
               if (COMP[i-1]==0 and COMP[i]==1): #low to high
                  t_LH.append(DT[i])
                  flag_LH =1
                  index_LH.append(i) 
               if (COMP[i-1]==1 and COMP[i]==0):#high to low
                  t_HL.append(DT[i])
                  flag_HL = 1
                  index_HL.append(i)
   
            if flag_HL ==1 and flag_LH ==1 and int(np.ceil(abs(t_LH[-1]-t_HL[-1])))<5*60:
                flag_HL =0
                flag_LH =0
                del t_LH[-1]
                del index_LH[-1]
                del t_HL[-1]
                del index_HL[-1]
            
            if int(np.ceil(DT[i]-DT[i-1]))>60*5 and flag_HL ==0 and flag_LH==0:  #and flag_training==1 case a
                flag_interrupt_a =1
     
            if int(np.ceil(DT[i]-DT[i-1]))>60*5 and flag_LH ==1 and flag_HL ==0:  # and flag_training==1 case b
                flag_LH=0
                del t_LH[-1]
                del index_LH[-1]
                flag_interrupt_b =1
       
            if flag_interrupt_b ==1 and flag_LH ==1 and flag_HL==0: # and flag_training==1:
                flag_LH =0
                flag_interrupt_b =0
                del t_LH[-1]
                del index_LH[-1]
                flag_LH_after_b =1

      
            if flag_interrupt_b ==1 and flag_LH==0 and flag_HL ==1: # and flag_training==1:
                flag_HL =0
                flag_interrupt_b =0
    
            if flag_interrupt_a ==1 and flag_LH==1 and flag_HL==0: #and flag_training==1:
                flag_LH=0
                flag_interrupt_a =0
                del t_LH[-1]
                del index_LH[-1]
                flag_LH_after_a =1
            if flag_LH_after_a ==1 and flag_HL==1: #and flag_training==1:
                flag_LH_after_a=0
                flag_HL=0
            if flag_LH_after_b==1 and flag_interrupt_a ==1 and flag_HL==0 and flag_LH==0: #and flag_training==1:
                flag_LH_after_b=0
        
            if flag_interrupt_a ==1 and flag_HL==1 and flag_LH==0: # and flag_training==1:
                if flag_interrupt_b==1:
                    flag_interrupt_b=0
                flag_HL=0
                flag_interrupt_a =0
        
            if flag_LH_after_b==1 and flag_HL==1 and flag_interrupt_a==0: #and flag_training==1:
                flag_LH_after_b=0 
                flag_HL=0 
       
            if flag_HL==1 and flag_LH==1: 
                if flag_training==1:
                   if  int(np.ceil(t_HL[-1]-t_LH[-1]))<600 or int(np.ceil(t_HL[-1]-t_LH[-1]))>3600: ## t_HL[-1]>t_LH[-1] and#and 60<(t_HL[len(t_HL)-1]-t_LH[len(t_LH)-1]): 
                       flag_HL=0
                       flag_LH=0
                       del t_LH[-1]
                       del index_LH[-1] 
                else:
                    if t_HL[-1]>t_LH[-1] and int(np.ceil(t_HL[-1]-t_LH[-1]))<120:
                        flag_HL=0
                        flag_LH=0
                        del t_LH[-1]
                        del index_LH[-1]
                        del t_HL[-1]
                        del index_HL[-1]
            
                   
              #
            if flag_msg==0  and flag_HL==0 and flag_training==0 and  flag_interrupt_b==0 and flag_LH_after_b==0 and flag_LH_after_a==0: 
               if COMP[i]==0 and flag_LH==0:            
                  if int(np.ceil(DT[i]-DT[i-1]))>60*2:
                      count_w_comp =0
                  else:
                      count_w_comp = count_w_comp+1
                  if (count_w_comp>420) and int(np.ceil(DT[i]-t_HL[-1]))>420:
                     print("Comp is Working more than_First",int(np.ceil(DT[i]-t_HL[-1])))
                     flag_msg = 1
                     T_start_msg.append(DT[i])
                     t_LH.append(DT[i])
                     index_LH.append(i)
                     flag_LH=0
                   

            if flag_msg==1 and int(np.ceil(DT[i]-t_LH[-1]))>10*60 and flag_LH==0 and flag_HL==0 and flag_training==0:
                t_HL.append(DT[i])
                index_HL.append(i)
                BIN,time_s,time_e,dur_low_c,dur_high_c = self.Feature_values(TP2,TP3,H1,Motor_current,Oil_temp,DV_Pre,Reservoirs,num_bin_inc,num_bin_dec,t_LH,t_HL,index_LH,index_HL)
                Bins.append(BIN)
                Time_Start.append(time_s)
                Time_end.append(time_e)
                IndexStart.append(index_HL[-2])
                IndexEnd.append(index_HL[-1])
                flag_msg=0
                count_w_comp=0

                
            if flag_msg==1 and flag_LH==1 and flag_HL==1 and flag_interrupt_a==0:
                if  int(np.ceil(abs(t_HL[-1]-t_LH[-1])))<60*5:
                    del t_LH[-1]
                    del index_LH[-1]
                    del t_HL[-1]
                    del index_HL[-1]
                    flag_HL=0
                    flag_LH=0
                else:
                    flag_msg=0
                    count_w_comp=0
                    del t_LH[-2]
                    del index_LH[-2]
                
            elif flag_msg==1 and flag_LH==0 and flag_HL==0 and int(np.ceil(DT[i]-DT[i-1]))>60*2:
                print("Comp Worked for more than_third", int(np.ceil(DT[i]-t_HL[-1])))
                count_w_comp=0
                flag_msg=0
                work_time_more420.append(int(np.ceil(DT[i]-t_HL[-1])))
                T_end_msg.append(DT[i])
                flag_interrupt_a=0
           #####
            if flag_LH==1 and flag_HL==0 and flag_training==0 and  flag_interrupt_b==0 and flag_LH_after_b==0 and flag_LH_after_a==0: 
              if COMP[i]==1  and TP3[i]-TP3[i-1]==0 and int(np.ceil(DT[i]-t_LH[-1]))>3000: 
                  print("Comp is not Working more than",int(np.ceil(DT[i]-t_HL[-1])))
                  t_HL.append(DT[i])
                  index_HL.append(i)
                  BIN,time_s,time_e,dur_low_c,dur_high_c = self.Feature_values(TP2,TP3,H1,Motor_current,Oil_temp,DV_Pre,Reservoirs,num_bin_inc,num_bin_dec,t_LH,t_HL,index_LH,index_HL)
                  Bins.append(BIN)
                  Time_Start.append(time_s)
                  Time_end.append(time_e)
                  IndexStart.append(index_HL[-2])
                  IndexEnd.append(index_HL[-1])
                  flag_LH=0
                  
                                  
            if flag_LH==0 and flag_HL==0 and flag_training==0:     
               if COMP[i]==1  and TP3[i]-TP3[i-1]==0 and int(np.ceil(DT[i]-t_HL[-1]))>3600:
                   t_LH.append(t_HL[-1]+420)
                   index_LH.append(index_HL[-1]+420)
                   t_HL.append(DT[i])
                   index_HL.append(i)
                   BIN,time_s,time_e,dur_low_c,dur_high_c = self.Feature_values(TP2,TP3,H1,Motor_current,Oil_temp,DV_Pre,Reservoirs,num_bin_inc,num_bin_dec,t_LH,t_HL,index_LH,index_HL)
                   Bins.append(BIN)
                   Time_Start.append(time_s)
                   Time_end.append(time_e)
                   IndexStart.append(index_HL[-2])
                   IndexEnd.append(index_HL[-1])
                   
            #####
            if flag_HL==1 and flag_LH==1: 

                if flag_msg==0 and int(np.ceil(t_LH[-1]-t_HL[-2]))<60: #and int(np.ceil(t_HL[-1]-t_LH[-2]))<5*60:
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
                elif flag_training==1 and t_LH[-1]> t_HL[-2] and int(np.ceil(t_LH[-1]-t_HL[-2]))>420:
                   flag_HL=0
                   flag_LH=0
                   del t_LH[-1]
                   del index_LH[-1]     
                else:
                     BIN,time_s,time_e,dur_low_c,dur_high_c = self.Feature_values(TP2,TP3,H1,Motor_current,Oil_temp,DV_Pre,Reservoirs,num_bin_inc,num_bin_dec,t_LH,t_HL,index_LH,index_HL)
                     Bins.append(BIN)
                     Time_Start.append(time_s)
                     Time_end.append(time_e)
                     dur_low_comp.append(dur_low_c)#.total_seconds())
                     dur_high_comp.append(dur_high_c)#.total_seconds())
                     IndexEnd.append(index_HL[-1])
                     IndexStart.append(index_HL[-2])
                     flag_LH=0
                     flag_HL=0  
                     
        return Bins,Time_Start,Time_end,IndexEnd,IndexStart



    def DigitalSensor_analysis(self,data,flag_training,IndexStart,IndexEnd):
        Xtrain=[]
        Xtest=[]
      #  self.ax1.set_title('Binary Code ' + str(single_date), fontsize=8)
        if len(IndexStart)>0:
            mydict = {

            }           
            for index in range(len(IndexStart)): 
                #  aux = ''
                mydict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}#
                for j in range(IndexStart[index],IndexEnd[index]):
                    aux = ''
                    c=0
                    for valu in data.iloc[j, 9:].values: # 9 is because the flowmeter data is removed from dataset
                        if valu>0:
                            #if c not in mydict.keys(): 
                                #   mydict[c] = 1
                                #  else:
                            mydict[c] += 1
                        c+=1


                x = []
                y = []
                for key in range(8): #sorted(mydict, key=keyfunction, reverse=True)[:10]:
                    x.append(int(key))
                    y.append(mydict[key])
                 #   print("key: "+str(key)+" value: "+str(mydict[key]))

                if flag_training==1:       
                    Xtrain.append(y)
                else:
                    Xtest.append(y)
        return Xtrain,Xtest
    

   
    def Add_Features(self,flag_training,X):

        if len(self.IndexStart)>0:

            if flag_training==1:       
                self.X_train = self.Bins
                self.X_train=np.append(self.X_train,X,axis=1)
            else:
                self.X_test = self.Bins
                self.X_test=np.append(self.X_test,X,axis=1)
           
    
    def Standardization(self,flag_training):
         scaler = StandardScaler().fit(self.X_train)
         if flag_training==1:
             self.X_train_full.append(self.X_train)
             self.X_train_scaled = scaler.transform(self.X_train)
         else:
             self.X_test_full.append(self.X_test)
             self.X_test_scaled = scaler.transform(self.X_test)
    def split_sequences(self,sequences, n_steps):
         X = list()   
         for i in range(len(sequences)):
		     # find the end of this pattern
             end_ix = i + n_steps
		    # check if we are beyond the dataset
             if end_ix > len(sequences):
                 break
             # gather input and output parts of the pattern
             seq_x = sequences[i:end_ix]
             X.append(seq_x)
         return np.array(X)
     
    def flatten(self,X):
        '''Input:a 3D array (sample x timesteps x features)   
        Output: flattened_X  A 2D array, sample x features
        '''
        flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1]-1), :]
        return(flattened_X)
    
    def simple_LPF_input(self,x,alpha):
        Y_LPF=[[0 for l in range(x.shape[1])] for y in range(x.shape[0])]
        Y_LPF[0]=x[0]
        for i in range(1,x.shape[0]):
            Y_LPF [i]=Y_LPF [i-1]+alpha*(x[i]-Y_LPF[i-1])
        self.x_filtered=Y_LPF

    def simple_LPF_output(self,x,alpha):
        Y_LPF=np.zeros([len(x)])
        Y_LPF[0]=0
        for i in range(1,len(x)):
            Y_LPF [i]=Y_LPF [i-1]+alpha*(x[i]-Y_LPF[i-1])
        self.y_filtered=Y_LPF     
        
    def evalution(self,flag_training):
         self.Standardization(flag_training)
         #scaler = StandardScaler().fit(self.X_train)
         #self.X_train_scaled = scaler.transform(self.X_train)
         
        # scaler = StandardScaler()
        # self.X_train_scaled = np.array(self.X_train).reshape(-1,1)
        # self.X_train_scaled = scaler.fit_transform(self.X_train_scaled)
          # self.X_train = self.X_train.flatten()
          #X_test = scaler.transform(X_test)   

          ############################# 
         opt=keras.optimizers.Adam()
         #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.01)#momentums = [0.0, 0.5, 0.9, 0.99]
         self.autoencoder.compile(optimizer=opt, loss="mse",metrics=["accuracy"])
         file_name = 'my_saved_model_new'
         tensorboard_callback = TensorBoard(log_dir="logs\\{}".format(file_name))
         ## (First, anaconda promot should go to this directory)(Type on anaconda promot) python -m tensorboard.main --logdir=logs/ --port=6007 (--port=.. for changing port number)
         
         # log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
         #tensorboard_callback = TensorBoard(log_dir="logs/{}".format(time()))
         #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)  
        ## tensorBoard --logdir=logs/
         history = self.autoencoder.fit(self.X_train_scaled, self.X_train_scaled,epochs=200, batch_size=100,
                validation_split=0.1,
                shuffle=False,
                #shuffle=True,
                #validation_data=(X_valid, X_valid),
                callbacks=[
                         keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, mode="min"),tensorboard_callback])
        # self.loss_training = history.history["loss"]
         
         #tf.summary.scaler
         #tf.summary.histogram
        # tf.summary.tensor
        
         plt.subplot(1,2,1)
         plt.plot(history.history["loss"], label="Training Loss")
         plt.plot(history.history["val_loss"], label="Validation Loss")
         plt.xlabel("Number of epoch")
         plt.ylabel("Loss")
         plt.legend()
         plt.show()
         
         plt.subplot(1,2,2)
         plt.plot(history.history["accuracy"], label="Training accuracy")
         plt.plot(history.history["val_accuracy"], label="Validation accuracy")
         plt.xlabel("Number of epoch")
         plt.ylabel("accuracy")
         plt.legend()
         plt.show()
         
         X_train_pred = self.autoencoder.predict(self.X_train_scaled)
         self.train_rmse_loss = np.sqrt(np.mean((X_train_pred - self.X_train_scaled)**2, axis=1))
         plt.hist(self.train_rmse_loss, bins=50)   
         threshold = np.max(self.train_rmse_loss)
         median = np.median(self.train_rmse_loss)
         upper_quartile = np.percentile(self.train_rmse_loss, 75)
         lower_quartile = np.percentile(self.train_rmse_loss, 25)
         iqr = upper_quartile - lower_quartile
         upper_whisker =self. train_rmse_loss[self.train_rmse_loss<=upper_quartile+1.5*iqr].max()
         lower_whisker = self.train_rmse_loss[self.train_rmse_loss>=lower_quartile-1.5*iqr].min()
         self.thr_boxplot.append(upper_quartile+3*iqr) #extreme outliers
         
    
    def evalution_LSTM(self,flag_training,encoding_dim,n_steps):
        self.Standardization(flag_training) 
        sequences = self.X_train_scaled
        self.X_train_scaled_LSTM= self.split_sequences(sequences,n_steps)
        ##############################
        #input_dim = self.X_train_scaled_LSTM.shape[2]
        #n_steps = self.X_train_scaled_LSTM.shape[1]
        #n_steps=X_train.shape[1], n_features=X_train.shape[2]
          ############################# 
        #self.X_train_scaled_LSTM,self.X_val_scaled_LSTM, self.X_train_scaled_LSTM, self.X_val_scaled_LSTM = train_test_split(self.X_train_scaled_LSTM, self.X_train_scaled_LSTM, test_size=0.1, random_state=0)  
          #################################
        #opt=keras.optimizers.Adam()
        opt=tf.keras.optimizers.Adam(learning_rate=1e-4)
        #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.01)#momentums = [0.0, 0.5, 0.9, 0.99]
        self.AELSTM.compile(optimizer=opt, loss="mse", metrics=['accuracy'])
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                         factor=0.5,
                                                         patience=10,
                                                         verbose=1,
                                                         min_delta=0.0001,
                                                         min_lr=1e-6,
                                                         mode='auto',
                                                         cooldown=5)
       # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
        EarlyStopping_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, mode="min")
        history_AELSTM = self.AELSTM.fit(self.X_train_scaled_LSTM, self.X_train_scaled_LSTM,epochs=200, batch_size=100,
                validation_split=0.1,
                shuffle=False,
                verbose=2,
                callbacks=[EarlyStopping_callback,reduce_lr])#
        
        plt.subplot(1,2,1)
        plt.plot(history_AELSTM.history["loss"], label="Training Loss")
        plt.plot(history_AELSTM.history["val_loss"], label="Validation Loss")
        plt.xlabel("Number of epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        
        plt.subplot(1,2,2)
        plt.plot(history_AELSTM.history["accuracy"], label="Training accuracy")
        plt.plot(history_AELSTM.history["val_accuracy"], label="Validation accuracy")
        plt.xlabel("Number of epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()
        X_train_pred = self.AELSTM.predict(self.X_train_scaled_LSTM)
        train_mse_loss = np.mean(np.power(self.flatten(X_train_pred) - self.flatten(self.X_train_scaled_LSTM), 2), axis=1)   
        self.recon_error_trainingData = train_mse_loss = (self.flatten(X_train_pred) - self.flatten(self.X_train_scaled_LSTM))   
        train_mae_loss = np.mean(self.flatten(X_train_pred) - self.flatten(self.X_train_scaled_LSTM), axis=1)
        threshold = np.max(train_mse_loss)
        median = np.median(train_mse_loss)
        upper_quartile = np.percentile(train_mse_loss, 75)
        lower_quartile = np.percentile(train_mse_loss, 25)
        iqr = upper_quartile - lower_quartile
        upper_whisker =train_mse_loss[train_mse_loss<=upper_quartile+1.5*iqr].max()
        lower_whisker = train_mse_loss[train_mse_loss>=lower_quartile-1.5*iqr].min()
        self.thr_boxplot_LSTM.append(upper_quartile+3*iqr) #  midle and 3 extreme outliers 
       
        #### validation data
        '''
        X_val_pred = self.AELSTM.predict(self.X_val_scaled_LSTM)
        val_mse_loss = np.mean(np.power(self.flatten(X_val_pred) - self.flatten(self.X_val_scaled_LSTM), 2), axis=1)   
        upper_quartile = np.percentile(val_mse_loss, 75)
        lower_quartile = np.percentile(val_mse_loss, 25)
        iqr = upper_quartile - lower_quartile
        self.thr_boxplot_LSTM.append(upper_quartile+1.5*iqr) #midle and 3 extreme outliers 
        '''
        ######
        train_mae_loss = np.mean(np.abs(self.flatten(X_train_pred) - self.flatten(self.X_train_scaled_LSTM)), axis=1)
        threshold = np.max(np.abs(train_mae_loss))
        #median = np.median(train_mae_loss)
        upper_quartile = np.percentile(train_mae_loss, 75)
        lower_quartile = np.percentile(train_mae_loss, 25)#
        iqr = upper_quartile - lower_quartile
        #upper_whisker =train_mae_loss[train_mae_loss<=upper_quartile+1.5*iqr].max()
        #lower_whisker = train_mae_loss[train_mae_loss>=lower_quartile-1.5*iqr].min()
        self.thr_boxplot_mae_LSTM.append(upper_quartile+3*iqr) #extreme outliers 
        self.thr_max_mae_LSTM.append(threshold)
        
    def evalution_TestData(self,flag_training,alpha_SLPF):
        self.Standardization(flag_training)
        X_test_pred = self.autoencoder.predict(self.X_test_scaled, verbose=0)
        recon_error=X_test_pred-self.X_test_scaled
        self.test_rmse_loss = np.sqrt(np.mean(((X_test_pred-self.X_test_scaled))**2, axis=1))#“axis 0” represents rows and “axis 1” represents columns.
        self.pred_y_test = [1 if e > self.thr_boxplot[-1] else 0 for e in self.test_rmse_loss]
        self.simple_LPF(self.pred_y_test,alpha_SLPF)
        

        self.Y_Test_pre.extend(self.pred_y_test)
        self.Y_Test_preLPF.extend(self.pred_y_test_simpleLPF)
        '''
        anomaly_testData =[]
        anomaly_testData['test'] =pd.DataFrame(self.Y_Test_preLPF)
        anomaly_testData['anomaly']= self.Y_Test_preLPF> 0.5
        anomalies = anomaly_testData.loc[anomaly_testData['anomaly']==True]
        sns.linplot(x=self.Time_StartCOMP, y=scaler.inverse.transform(anomaly_testData['test']))
        sns.scaterplot(x=self.Time_StartCOMP, y=scaler.inverse.transform(anomalies['test']),color='red')
        '''
       
    def evalution_TestData_LSTM(self,flag_training,alpha_SLPF,n_steps):
        self.Standardization(flag_training)
        sequences = self.X_test_scaled
        self.X_test_scaled_LSTM= self.split_sequences(sequences,n_steps)
        self.X_test_pred_LSTM = self.AELSTM.predict(self.X_test_scaled_LSTM, verbose=2)
        self.recon_error_LSTM=np.abs(self.flatten(self.X_test_pred_LSTM)-self.flatten(self.X_test_scaled_LSTM))
        self.test_mse_loss_LSTM = np.sqrt(np.mean(np.power(self.flatten(self.X_test_pred_LSTM) - self.flatten(self.X_test_scaled_LSTM), 2), axis=1))
        #“axis 0” represents rows and “axis 1” represents columns.
        self.Y_Test_pre_LSTM = [1 if e > self.thr_boxplot_LSTM[-1] else 0 for e in self.test_mse_loss_LSTM]
       
        self.test_mae_loss_LSTM = np.mean(abs(self.flatten(self.X_test_pred_LSTM) - self.flatten(self.X_test_scaled_LSTM)), axis=1)
        self.Y_Test_pre_LSTM_mae = [1 if e > self.thr_max_mae_LSTM[-1] else 0 for e in np.abs(self.test_mae_loss_LSTM)]
        
       # anom_index = np.where(self.Y_Test_pre_LSTM==1)
       # values = self.Y_Test_pre_LSTM[anom_index]
        self.simple_LPF_output(self.Y_Test_pre_LSTM,alpha_SLPF)       
        self.Y_Test_preLPF_LSTM = self.y_filtered
        
        self.simple_LPF_output(self.Y_Test_pre_LSTM_mae,alpha_SLPF)       
        self.Y_Test_preLPF_LSTM_mae = self.y_filtered
        
       
    def __init__(self, parent=None):

        self.X_train_full = []
        self.X_test_full = []
        self.thr_boxplot =[]
        self.thr_boxplot_LSTM =[]
        self.thr_boxplot_mae_LSTM =[]
        self.thr_max_mae_LSTM =[]
        self.Y_Test_preLPF=[]
        self.Y_Test_pre=[]
        self.pred_test_data=[]
        self.X_test_INPUT=[]
        self.rmse_OUTPUT=[]

        self.x=[]
        self.X_train = []
        self.X_test = []
            
        flag_training =1
        train_data, test_data= self.load_Data_csvfile()
        
        Bins,Time_Start,Time_end,IndexEnd,IndexStart = self.Find_BestDuration(train_data,flag_training)
        Xtrain,_ =self.DigitalSensor_analysis(train_data,flag_training,IndexStart,IndexEnd)

        #self.Add_Features(flag_training,Xtrain)
        X_ttrain=np.append(Bins,Xtrain,axis=1)
        self.X_train=np.array(X_ttrain)
        
        _,input_dim = np.shape(self.X_train)
        encoding_dim = int(input_dim/2)
        n_features = input_dim
        n_steps = 3
        #self.model_DeepNetwor(input_dim,encoding_dim)
        #self.model_SimpleNetwork(input_dim,encoding_dim)
        #self.model_Deep_AELSTM(n_features,encoding_dim,n_steps) 
        self.model_Deep_AELSTM_2(n_features,encoding_dim,n_steps) 
        #self.evalution(flag_training)
        self.evalution_LSTM(flag_training,encoding_dim,n_steps)
            #########################
                ##Test
        flag_training=0
        alpha_SLPF = 0.07
        test_data = test_data.reset_index(drop=True)
        Bins,Time_Start,Time_end,IndexEnd,IndexStart = self.Find_BestDuration(test_data,flag_training)
        Data5=pd.DataFrame(Time_Start)
        Data5.to_csv('Time_StartTrainig_allsens_Staticlstm_metropt1.csv') 
        Data4=pd.DataFrame(Time_end)
        Data4.to_csv('Time_EndTraining_allsens_Staticlstm_metropt1.csv') 
        _, Xtest = self.DigitalSensor_analysis(test_data,flag_training,IndexStart,IndexEnd)
        X_ttest= np.append(Bins,Xtest,axis=1)
        #self.Add_Features(flag_training,Xtest)
        self.X_test=np.array(X_ttest)
        #self.evalution_TestData(flag_training,alpha_SLPF)
        self.evalution_TestData_LSTM(flag_training,alpha_SLPF,n_steps)

        
        Data8 = pd.DataFrame(self.recon_error_trainingData)
        Data8.to_csv('recon_error_trainingData_LSTM_allsens_staticlstm_metropt1.csv')
        
        Data7=pd.DataFrame(self.recon_error_LSTM)
        Data7.to_csv('recon_error_LSTM_allsens_staticlstm_metropt1.csv')
        
        Data6=pd.DataFrame(self.thr_boxplot_LSTM)
        Data6.to_csv('Threshold_Boxplot_allsens_Staticlstm_metropt1.csv')  
        
        #Data5=pd.DataFrame(self.Time_Start)
       # Data5.to_csv('Time_StartTest_allsens_Staticlstm_metropt1.csv') 

        #Data4=pd.DataFrame(self.Time_end)
        #Data4.to_csv('Time_EndTest_allsens_Staticlstm_metropt1.csv') 
        
        Data3=pd.DataFrame(np.array(self.X_test))
        Data3.to_csv('X_test(INPUT)_allsens_Staticlstm_metropt1.csv') 
        
        Data2=pd.DataFrame(np.array(self.test_mse_loss_LSTM))
        Data2.to_csv('RMSError(OUTPUT)_LSTM_allsens_Staticlstm_metropt1.csv') 
        
        Data1= pd.DataFrame(self.Y_Test_pre_LSTM)
        Data1.to_csv('Y_testPredict_allsens_Staticlstm_metropt1.csv')
        
        

if __name__ == '__main__':
    zc = ZombieScript()

#Data=pd.DataFrame(zc.X_train)
#Data.to_csv('X_train.csv') 