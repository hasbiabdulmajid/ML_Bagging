#!/usr/bin/env python
# coding: utf-8

# In[4]:


import csv
import numpy as np
import pandas as pd
import time
import random
from statistics import mode
from sklearn.naive_bayes import GaussianNB


# In[5]:


d_Train = np.genfromtxt('TrainsetTugas4ML.csv', delimiter=',', dtype=None, skip_header=1, names=['X1','X2','Class'])
d_Test = np.genfromtxt('TestsetTugas4ML.csv', delimiter=',', dtype=None, skip_header=1)


# In[12]:


# menambahkan label pada data
def addLabel(data):
    kolom = ['X1','X2', 'Class']
    index = []
    for i in range(len(data)):
        index.append(str(i))
    temp = pd.DataFrame(data, columns=kolom,index = index)
    return temp


# In[43]:


# Fungsi untuk melakukan proses pembuatan model bootstrap dan melakukan prediksi menggunakan naivebayes
def bootstrapFunc(iteration):
    for i in range(iteration):
        tabel = dataLatih.sample(190)
        tmp = GaussianNB()
        tmp.fit(dataLatih[atribut].values,dataLatih['Class'])
        tmp_prediksi = tmp.predict(dataUji[atribut])
        prediksi.append(tmp_prediksi)
        


# In[32]:


def tambahElemen(x,y):
    for j in range (300) :
        elemen = prediksi[j][y]
        x.append(elemen)


# In[44]:


dataLatih = addLabel(d_Train)
dataUji = addLabel(d_Test)
bootstrap = 300
prediksi = []
atribut = ['X1','X2']
bootstrapFunc(bootstrap)
# print(prediksi)


# In[34]:


pred_final = [] # menampung hasil prediksi final
tampung = [] # untunk menampung prediksi
for i in range(len(prediksi[1])) :
        tampung.clear()
        tambahElemen(tampung,i)
        modus = mode(tampung) #voting
        pred_final.append(modus)
print(pred_final)
pd.DataFrame(pred_final).to_csv("TebakanTugas4.csv",index=False,header=False) #export
            

