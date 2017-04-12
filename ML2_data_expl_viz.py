################################################
#data exploration & viz
################################################

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 20:17:39 2017

@author: Jonas
"""





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



#load data
ibm = pd.read_csv('IBM_stock.csv', index_col=0)
aapl = pd.read_csv('Apple_stock.csv', index_col=0)
fb = pd.read_csv('Facebook_stock.csv', index_col=0)
googl = pd.read_csv('Google_stock.csv', index_col=0)



#show basic summaries

print(ibm.describe())
print(aapl.describe())
print(fb.describe())
print(googl.describe())




#data visualization
#---------------------------------------------------------------

#timeseries plots

fig, (ax1,ax2,ax3,ax4) = plt.subplots(ncols=4,figsize=(32, 8))
sns.tsplot(ibm.Open, ax=ax1)
sns.tsplot(aapl.Open, ax=ax2)
sns.tsplot(fb.Open, ax=ax3)
sns.tsplot(googl.Open, ax=ax4)

ax1.set_title('IBM')
ax2.set_title('Apple')
ax3.set_title('Facebook')
ax4.set_title('Google')



#distributions

fig, (ax1,ax2,ax3,ax4) = plt.subplots(ncols=4,figsize=(32, 8))
sns.distplot(ibm.Open, ax=ax1)
sns.distplot(aapl.Open, ax=ax2)
sns.distplot(fb.Open, ax=ax3)
sns.distplot(googl.Open, ax=ax4)

ax1.set_title('IBM')
ax2.set_title('Apple')
ax3.set_title('Facebook')
ax4.set_title('Google')



#variance and range

print('--IBM--')
print(ibm.var())
print('mean diff. of High/Low  ',ibm.High.mean()-ibm.Low.mean())
print('--Apple--')
print(aapl.var())
print('mean diff. of High/Low  ',aapl.High.mean()-aapl.Low.mean())
print('--Facebook--')
print(fb.var())
print('mean diff. of High/Low  ',fb.High.mean()-fb.Low.mean())
print('--Google--')
print(googl.var())
print('mean diff. of High/Low  ',googl.High.mean()-googl.Low.mean())









