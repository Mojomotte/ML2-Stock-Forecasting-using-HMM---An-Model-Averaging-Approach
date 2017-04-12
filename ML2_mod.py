###############################################
#Model
###############################################


# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 11:21:26 2017

@author: Jonas
"""


import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM as ghmm
import math
import copy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 




#FUNCTIONS
#############################################

#create functions for calculation of information criteria
class model_fit_measure:
     
     def AIC(L,k):
         return(-2*math.log(abs(L))+2*k)
         
     def BIC(L,k,M):
         return(-2*math.log(abs(L))+k*math.log(M))




#basic hmm training function
#-----------------------------------------------------------
def build_hmm_model(data, states, start_prob=None, mu=None, sigma=None, 
                    ex = [], max_iterations=1000, EM_threshold=0.01, 
                    conv=False, show=True):

    if show == True:
        print('Fitting and decoding... ', end='')
    
    #create basic model
    model = ghmm(n_components = states,
                 covariance_type='diag',
                 n_iter = max_iterations,
                 tol = EM_threshold)
    
    
    if 'sp' in ex:
        model.startprob_ = start_prob
    
    if 'm' in ex:    
        model.means_prior = mu
        
    if 's' in ex:
        model.covars_prior = sigma
    
        
    #fit model to data
    model.fit(data)
    
    #convergence information
    if conv == True:
        print('\n--Convergence--')
        print(model.monitor_.converged)
        print(model.monitor_) 
    
    if show == True:
        print('done')
    
    
    #calculate likelihood of observations given model
    mod_likelihood = model.score(data)
    
    #calculate information criteria
    k = states*8  ##mu and sigma for 4 dim
    n = len(data.index)  ##number of observations
    
    aic = model_fit_measure.AIC(mod_likelihood,k)
    bic = model_fit_measure.BIC(mod_likelihood,k,n)
    
    
    #format output as python data type "dictionary"
    out = {'model' : model,
           'likelihood' : mod_likelihood,
           'aic' : aic,
           'bic' : bic
           }   
        
    
    return(out)


    
    
    
#calculates parameters for future iterations by one initial run of hmm
#-----------------------------------------------------------    
def hmm_init(data):                           
    
    out = []
    
    print('Initializing...')
    
    for i in range(2,6):
    
        #build model        
        model = build_hmm_model(data=data, states=i, show=False)
        
        #get parameters
        mod_0 = model['model']        
        start_prob_0 = mod_0.startprob_
        mu_0 = mod_0.means_
        sigma_0 = mod_0.covars_
        
        #format output as list
        lambda_0 = [start_prob_0, mu_0, sigma_0]
        out.append( lambda_0)
        
    return(out)       
        

    
    
             
#calculates the predicted difference of todays value and 
#the next-day value of the Close Price
#-----------------------------------------------------------
def predict_diff(T, data_train, data_predict, states, 
                 start_prob, mu, sigma, conv=False):
    
    
    t_init = len(data_train.index)  
    ##index of the last entry of the training data
    
    t0 = t_init-1  
    ##index of the 2nd last entry of the training data --> point to start prediction
    
    
    data_t = data_train.iloc[(t_init-T):t_init]
    

    #build hmm model with initialization parameters             
    model = build_hmm_model(data=data_t, states=states, 
                            start_prob=start_prob, mu=mu, 
                            sigma=sigma, ex=['sp','m','s'], show=True)
            
            
    mod = model['model']
    L_target = model['likelihood']
    
    #convergence information
    if conv == True:
        conv = mod.monitor_.converged
        print('converged: ', end='')
        print(conv)
    
    
    val = []
    cond = []
    
    #use interval of length T and move it one datapoint back in each run
    for i in range(1,(t0-T)):
        t_b = (t0-i)  ##define borders of interval T
        t_a = (t_b-T)
        seq = data_train.iloc[t_a:t_b]                
        
        likelihood = mod.score(seq)  
        ##predict likelihood of the model for the "new" data
        
        l_diff = (likelihood-L_target)  
        ##calculate diff of likelihood of "new" data and actual data (t0)
        
        cond.append( abs(l_diff))
        
        sign = np.sign(l_diff)  ##get the sign of the likelihood diff
        
        #calculate diff of Close Price of the current day and its successor
        val.append((data_predict.Close.iloc[(t_b+1)] - data_predict.Close.iloc[(t_b)])*sign)
     
        
    start_prob = mod.startprob_
    mu = mod.means_
    sigma = mod.covars_
    
    #format output
    out = {'res' : pd.DataFrame({'cond':cond,
                  'val': val}),
           'aic' : model['aic'],
           'bic' : model['bic'],
           'sp' : start_prob,
           'mu' : mu,
           'sigma' : sigma}    
    
    return(out)
    
    
    

    

#DATA
#####################################################    

    
    
#load data  (seperate runs for each import data file)
#----------------------------------------------------
data = pd.read_csv('IBM_stock.csv', index_col=0)
#data = pd.read_csv('Apple_stock.csv', index_col=0)
#data = pd.read_csv('Facebook_stock.csv', index_col=0)
#data = pd.read_csv('Google_stock.csv', index_col=0)


''' --> use August 2014 - 2016 for training, then predict Sep.-Nov. 2016 '''

#data used for prediction 
#----------------------------------------
#only use Open, Close, High, Low -price
data_predict = data.iloc[1:,0:4]



#data used for training: use 1st differences
#------------------------------------------------
data_train = data

#calculate differences
data_train.Open = data.Open.diff()
data_train.High = data.High.diff()
data_train.Low = data.Low.diff()
data_train.Close = data.Close.diff()

#only use Open, Close, High, Low -price
data_train = data_train.iloc[1:,0:4]




    

#CALCULATIONS
#####################################################    
 

#walking interval length    
T = 100

#get complete initial training data (August 2014 - 2016)
data_init = data_train[data_train.index < '2016-09-01']



#walking index 
t_walk = len(data_init.index)-1
   

#train initial hmm on full initial training set to get 
#good initial par for algorithm
res = hmm_init(data_init)

#set initial start probabilities to uniform dist
for p in range(0,4):
    res[p][0] = [(1/(p+2))]*(p+2)


#create df
final_diff = pd.DataFrame(columns=('diff1', 'diff2', 'diff3'))


#main calculation loop
#--------------------------------------------------------------------
for q in range(0,(len(data_predict.index)-len(data_init.index))):
    
    print()
    print('LOOP {}'.format((q+1)))
    #print(data_predict.index[t_walk+1])
    
    #data for training model and predict data points    
    data_t = data_train.iloc[0:t_walk]
    data_pred = data_predict.iloc[0:t_walk]
    
    
    pred = []
    
    #train 4 hmms with 2-5 states and predict with each
    for i in range(0,4):
        
        #get initial hmm parameters
        sp = res[i][0]
        mu = res[i][1]
        s = res[i][2]
        
        #format covariance values for input
        sigma = []
        for j in range(0,len(s)):
            sigma.append(np.diag(s[j]))
            
        #number of states
        st = i+2
        
        #calculate differences
        diff_obj = predict_diff(T=T, data_train=data_t, 
                                data_predict=data_pred, 
                                states=st, start_prob=sp, 
                                mu=mu, sigma=sigma, conv=True)
        
        #save parameters of this model as init par for 
        #successing model (successive update of init par)
        res[i][0] = diff_obj['sp']
        res[i][1] = diff_obj['mu']
        res[i][2] = diff_obj['sigma']
        
        #get values of information criteria and the list of 
        #predicted diff with corresponding likelihood diff
        aic = diff_obj['aic']
        bic = diff_obj['bic']
        res_list = diff_obj['res']
        
        #get diff that is most likely
        diff_pred_idx = res_list.idxmin(0)
        diff_pred = res_list.iloc[diff_pred_idx['cond']]
        
        #save result in multi dimensional list
        pred.append([aic,bic,diff_pred])
    
    #update walking index
    t_walk = (t_walk+1)
    
    
    
    
    
    #final results for predicted differences (different averaging methods)
    #--------------------------------------------------------------------------    
    

    #METHOD 1    
    #average value of model with best AIC and model with best BIC 50/50
    
    #read out results from list and create df
    x_aic = []
    x_bic = []
    x_cond = []
    x_val = []
    
    for k in range(0,4):
        x_aic.append(pred[k][0])
        x_bic.append(pred[k][1])
        x_cond.append(pred[k][2]['cond'])
        x_val.append(pred[k][2]['val'])
    
    cal = pd.DataFrame({'aic':x_aic,
                        'bic':x_bic,
                        'cond':x_cond,
                        'val':x_val})
    
    #calculate prediction as average of value with 
    #best AIC and value with best BIC
    final1 = ((cal.iloc[cal.idxmin(0)['aic']]['val']+
                        cal.iloc[cal.idxmin(0)['bic']]['val'])*0.5)
    
    
    
        
    
    #METHOD 2
    #total model averaging 
    '''
    #calculate weights by BIC and AIC "inversely" normalized by their totals
    #(since small aic or bic ist better than high), 
    #each multiplied by cond (=likelihood) and normalized again for all models 
    #and then calculate weighted average of predicted differences
    #finally average the BIC and AIC result values 50/50
    '''    
    
    pred2 = copy.deepcopy(pred)
    
    
    #if the abs difference of the likelihoods is smaller than 1, 
    #it will be set to 1 to avoid to much instability when multiplying
    for k in range(0,4):
        if pred2[k][2]['cond'] < 1:
            pred2[k][2]['cond'] = 1
        else:
            pred2[k][2]['cond'] = pred2[k][2]['cond']
    
    
    #calculate 1st totals
    aic_tot1 = 0
    bic_tot1 = 0
    cond1 = 0
    
    for k in range(0,4):
        aic_tot1 = (aic_tot1 + pred2[k][0])
        bic_tot1 = (bic_tot1 + pred2[k][1])
        cond1 = (cond1 + pred2[k][2]['cond'])
    
    for k in range(0,4):
        pred2[k][0] = (aic_tot1/pred2[k][0])
        pred2[k][1] = (bic_tot1/pred2[k][1])
        pred2[k][2]['cond'] = (cond1/pred2[k][2]['cond'])
    
    #multiply by likelihood difference
    for k in range(0,4):
        pred2[k][0] = (pred2[k][0]*pred2[k][2]['cond'])
        pred2[k][1] = (pred2[k][1]*pred2[k][2]['cond'])
    
    #calculate 2nd totals
    aic_tot2 = 0
    bic_tot2 = 0
    
    for k in range(0,4):
        aic_tot2 = (aic_tot2 + pred2[k][0])
        bic_tot2 = (bic_tot2 + pred2[k][1])
    
    for k in range(0,4):
        pred2[k][0] = (pred2[k][0]/aic_tot2)
        pred2[k][1] = (pred2[k][1]/bic_tot2)
    
    
    #calculate final value
    final2 = 0
    
    for k in range(0,4):
        final2 = final2+((pred2[k][0]+pred2[k][1])*0.5*pred2[k][2]['val'])
    
    
    
    #METHOD 3
    #selected average
    '''
    #choose only values that are "close enough" to the 
    #value the predicition is for
    #--> small enough cond (=diff of likelihood under the model)
    #calculate weights by BIC and AIC "inversely" normalized by their totals
    #(since small aic or bic ist better than high), 
    #and then calculate weighted average of predicted differences
    #finally average the BIC and AIC result values 50/50
    '''
    
    pred3 = copy.deepcopy(pred)
    
    #select values that are "close enough" 
    #--> within 1.25 times the mean of cond
    cond_max = 0
    
    for k in range(0,4):
        cond_max = (cond_max + pred3[k][2]['cond'])
    cond_max = 0.3125*cond_max ## mean*1.25
    
    condit = len(pred3)
    k = 0
    while k < condit:
        if pred3[k][2]['cond'] > cond_max:
            del pred3[k]
            condit = (condit-1)
        else:
            k += 1
    
    n = len(pred3)
    
    #calculate 1st totals
    aic_tot1 = 0
    bic_tot1 = 0
    
    
    for k in range(0,n):
        aic_tot1 = (aic_tot1 + pred3[k][0])
        bic_tot1 = (bic_tot1 + pred3[k][1])
    
        
    for k in range(0,n):
        pred3[k][0] = (aic_tot1/pred3[k][0])
        pred3[k][1] = (bic_tot1/pred3[k][1])
    
        
    #calculate 2nd totals
    aic_tot2 = 0
    bic_tot2 = 0
    
    for k in range(0,n):
        aic_tot2 = (aic_tot2 + pred3[k][0])
        bic_tot2 = (bic_tot2 + pred3[k][1])
    
    for k in range(0,n):
        pred3[k][0] = (pred3[k][0]/aic_tot2)
        pred3[k][1] = (pred3[k][1]/bic_tot2)
    
    #calculate final value    
    final3 = 0
    
    for k in range(0,n):
        final3 = final3+((pred3[k][0]+pred3[k][1])*0.5*pred3[k][2]['val'])
    
    
    #save all final values of the 3 methods in a list
    final = [final1,final2,final3]
    
    #and add it to a df
    final_diff.loc[q] = final
    
    print('FINISHED')
    
    


    
#calculate the predicted close price for the next day by 
#adding the predicted diff to the close price of the current day
final_df = pd.DataFrame(columns=('real','pred1', 'pred2', 'pred3'))
n = len(data_predict.index) - 64

for j in range(0,63):
    
    r = data_predict.iloc[(n+j+1)].Close
    f1 = data_predict.iloc[(n+j)].Close + final_diff.diff1.iloc[j]
    f2 = data_predict.iloc[(n+j)].Close + final_diff.diff2.iloc[j]
    f3 = data_predict.iloc[(n+j)].Close + final_diff.diff3.iloc[j]
    
    add = [r,f1,f2,f3]
    final_df.loc[j] = add
    



    
#export resulting df as .csv file for further work
#corresponding to import data file

final_df.to_csv('IBM_pred.csv', sep=',')
#final_df.to_csv('Apple_pred.csv', sep=',')
#final_df.to_csv('Facebook_pred.csv', sep=',')
#final_df.to_csv('Google_pred.csv', sep=',')




