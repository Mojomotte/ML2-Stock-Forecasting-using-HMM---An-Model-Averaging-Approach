###############################################
#model evaluation & result viz
###############################################

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 13:21:14 2017

@author: Jonas
"""


import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt



#load data
data = pd.read_csv('IBM_pred.csv', index_col=0)
#data = pd.read_csv('Apple_pred.csv', index_col=0)
#data = pd.read_csv('Facebook_pred.csv', index_col=0)
#data = pd.read_csv('Google_pred.csv', index_col=0)




#define function to calculate RMSE
def rmse(prediction, real):
    return np.sqrt(((prediction-real)**2).mean())


#format data for calculation
tar = np.array(data.real[:])
p1 = np.array(data.pred1[:])
p2 = np.array(data.pred2[:])
p3 = np.array(data.pred3[:])


#calculate RMSE, rounded to 4 digits
r1 = round(rmse(p1,tar), 4)
r2 = round(rmse(p2,tar), 4)
r3 = round(rmse(p3,tar), 4)


print()
print('RMSE results:')
print('Method 1: {}'.format(r1))
print('Method 2: {}'.format(r2))
print('Method 3: {}'.format(r3))



#plot time series of real values and the predicted values


#method 1
fig1, ax1 = plt.subplots(ncols=1,figsize=(15, 8))
data.loc[:,['real','pred1']].plot(colormap='bwr', ax=ax1, 
xticks=pd.Series(range(0,66,5)), xlim=[-2,64], marker='o')
ax1.set_title('Method 1', fontsize=20)
ax1.set_xlabel('Time Stamps (Trading-days)', fontsize=16)
ax1.set_ylabel('Close Price', fontsize=16)
ax1.legend(fontsize=15, loc=0)
ax1.grid()


#method 2
fig2, ax2 = plt.subplots(ncols=1,figsize=(15, 8))
data.loc[:,['real','pred2']].plot(colormap='bwr', ax=ax2, 
xticks=pd.Series(range(0,66,5)), xlim=[-2,64], marker='o')
ax2.set_title('Method 2', fontsize=20)
ax2.set_xlabel('Time Stamps (Trading-days)', fontsize=16)
ax2.set_ylabel('Close Price', fontsize=16)
ax2.legend(fontsize=15, loc=0)
ax2.grid()


#method 3
fig3, ax3 = plt.subplots(ncols=1,figsize=(15, 8))
data.loc[:,['real','pred3']].plot(colormap='bwr', ax=ax3, 
xticks=pd.Series(range(0,66,5)), xlim=[-2,64], marker='o')
ax3.set_title('Method 3', fontsize=20)
ax3.set_xlabel('Time Stamps (Trading-days)', fontsize=16)
ax3.set_ylabel('Close Price', fontsize=16)
ax3.legend(fontsize=15, loc=0)
ax3.grid()




#variance of real data
print()
print('Variance of real data: ')
print(round(data.loc[:,'real'].var(), 4))


   
    
    
