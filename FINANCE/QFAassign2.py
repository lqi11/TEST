#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 18:10:24 2019

@author: linqi
"""

import fix_yahoo_finance as yf
start = '2007-12-01'
end = '2018-11-30'
tickers = ['IBM','GOOGL','AMZN','KO','MSFT']
data = yf.download(tickers, start,end)

#get the monthly data, get the last day data in a month
pmo = data['Adj Close'].resample('M').last()
import numpy as np
import pandas as pd
import datetime 
import matplotlib.pyplot as plt
#holding period return
rmo = pmo[1:len(pmo)].values/pmo[0:len(pmo)-1]
rmo.index = np.arange(1, len(rmo) + 1)
import os
rf = open(os.path.expanduser("~/Documents/2019 Spring/QFA/risk free.csv"))#data path of risk free data
risk_free = pd.read_csv(rf)
risk_free = risk_free.iloc[11:,1]
excess_return = rmo.subtract(risk_free, axis = 0) 
excess_return = excess_return.iloc[11:,:]
mer = open(os.path.expanduser("~/Documents/2019 Spring/QFA/market excess return.csv"))
market_excess_return = pd.read_csv(mer)
market_excess_return = market_excess_return.iloc[:,1]
market_excess_return = market_excess_return[:-1]

market_excess_return = pd.DataFrame({'intercept': np.ones(131), 'Rm':market_excess_return})
market_excess_return = market_excess_return.as_matrix()
excess_return  = rmo.as_matrix()
market_excess_return = np.matrix(market_excess_return)
excess_return = np.matrix(excess_return)

[ai, bi] =((market_excess_return.T)*market_excess_return)**(-1)*(market_excess_return.T)*(excess_return)
ai = ai.T
bi= bi.T
#least abosulote deviation
#1.i = 1 j = 2, two layer loop to experience all point combination
i = 0

n = len(rmo)
beta = np.zeros((2,5))#the best beta combination
sum0 = 2**32 -1 #initionalize the first sumation 
while i < (n - 1):
    j = i + 1
    while j < n:
        #put the points into matrix
        y=np.concatenate((excess_return[i],excess_return[j]))
        if np.allclose(market_excess_return[i],market_excess_return[j]):
            break
        
        x=np.concatenate((market_excess_return[i],market_excess_return[j]))
        #beta1 = np.concatenate((beta[i], beta[j]))
        #excess_return1 = np.concatenate((excess_return[i], excess_return[j]))
        #market_excess_return1 = [market_excess_return[i], market_excess_return[j]]
        beta1 = x**(-1)*y#(2*2) * (2*5)=(2*5)
        # make a comparision, find the smallest abosulate
        yhat = market_excess_return*beta1#(n*2)*(2*5)=(n*5)
        sum1 = (excess_return - yhat).sum()
        if abs(sum1) < sum0:
            sum0 = abs(sum1)
            beta = beta1#(2*5)
        #if sum1 is less than sum0, use sum1 replace sum0
        j = j + 1
    i = i + 1
beta
#2.use the two point (1) to get a group of beta
#3. get y\hat use y\hat = x*beta
#4. get the sumation of all the (y - y\hat)
#5. repeat all the 2-4
# feel the beta is not that right, because it shouldn't be all less than -1

#shrinkage
#how to get betabar
# i plan to use (0-65) (65-130) to get two ybar, and x bar
#so that we can get ybar, xbar matrix to get betabar
ybar1 = excess_return[0:65].mean(0)
xbar1 = market_excess_return[0:65].mean(0)
ybar2 = excess_return[65:130].mean(0)
xbar2 = market_excess_return[65:130].mean(0)
ybar = np.concatenate((ybar1, ybar2))
xbar = np.concatenate((xbar1, xbar2))
betabar =xbar**(-1)*ybar# (2*2)*(2*5) = (2*5)
i = 0
v = np.zeros([1,5])
while i < len(bi):
    v[0,i] = 1/5 * (bi[i] - betabar[1,i])*(bi[i] - betabar[1,i])
    i = i + 1
    
alpha = 1-((5-3)/5) 
betaJS= betabar + alpha*(np.concatenate((ai.T,bi.T))-betabar) # (2*5)


#Bayesian approach 
#since Vasicek suggest use cross_sectional so i will use average 
#OLS beta across the five equity
#mean(ai) to get alpha bar mean(bi) to get beta bar
#mean(excess_return) to get a (n*1) r column
# market_excess_return is already a one column 
alphabar = np.mean(ai)
betabar1 = np.mean(bi) # call it betabar1 to diffirenciate betabar
rbar = np.mean(excess_return, axis = 1)#mean of excess_return
ssquare = (1/(n-2)) * (np.square(rbar - alphabar - np.multiply(betabar1,market_excess_return[:,1])).sum())
ssquare = np.divide(ssquare,v)
i = 0 


sigmasquare = np.zeros([1,5])
i = 0
while i < len(bi):
    sigmasquare[0,i] = (1/(n-2)) * (np.square(excess_return[:,i] - ai[i] - np.multiply(bi[i],market_excess_return[:,1])).sum())
    i = i + 1
sigmasquare = np.divide(sigmasquare,v)
w = 1/ssquare * (1/(1/ssquare + 1/sigmasquare))
i = 0
Ebeta = np.zeros([1,5])
while i < len(bi):
    Ebeta[0,i] = (1-w[0,i])*bi[i] + w[0,i] * betabar1
    i = i + 1
Ebeta


#question 3, 
# for OLS E(beta) is the same as bi
# sigma

BsigmaOLS = np.zeros([1,5])
varianceOLS = np.zeros([1,5])
i = 0
while i < len(bi):
    BsigmaOLS[0,i] = (1/(n-2)) * (np.square(excess_return[:,1] - ai[i] - np.multiply(bi[i],market_excess_return[:,1])).sum()) #si of OLS sigma
    BsigmaOLS[0,i] = BsigmaOLS[0,i]/v[0,i]
    varianceOLS[0,i] = 1/(1/BsigmaOLS[0,i] + 1/sigmasquare[0,i])
    i = i + 1
BsigmaOLS  
#draw the five baysian 
import matplotlib.mlab as mlab
import math
sigma = np.sqrt(varianceOLS)
#i = 0
#while i < len(bi):
# draw a (betabar, si)

i = 0
while i < len(bi):
    x = np.linspace(np.asscalar(bi[i] - 3*sigma[0,i]), np.asscalar(bi[i] + 3*sigma[0,i]), 100)
    
    plt.plot(x, mlab.normpdf(x, np.asscalar(bi[i]), np.asscalar(sigma[0,i])))   
    i = i + 1    
plt.show()

# use LAD 
i = 0
ssquareLAD = np.zeros([1,5])
Ebeta = np.zeros([1,5])
varianceLAD = np.zeros([1,5])
while i < len(bi):
    ssquareLAD[0,i] = (1/(n - 2)) * (np.square(excess_return[i,:] - beta[0,i] - np.multiply(beta[1,i],market_excess_return[:,1])).sum())
    ssquareLAD[0,i] = np.divide(ssquareLAD[0,i], v[0,i])
    ssquareLADsigma = np.sqrt(ssquareLAD)
    # draw of original LAD
    xLAD = np.linspace(np.asscalar(beta[0,i] - 3 * ssquareLADsigma[0,i]), np.asscalar(beta[0,i] + 3 * ssquareLADsigma[0,i]), 100)
    plt.plot(xLAD,mlab.normpdf(xLAD,np.asscalar(beta[0,i]), np.asscalar(ssquareLADsigma[0,i])),color = 'k')
    w[0,i] = 1/ssquareLAD[0,i] * (1/(1/ssquareLAD[0,i] + 1/sigmasquare[0,i]))
    Ebeta[0,i] = (1-w[0,i])*bi[i] + w[0,i]*beta[0,i]
    varianceLAD[0,i] = 1/(1/ssquareLAD[0,i] + 1/sigmasquare[0,i])
    sigmaLAD = np.sqrt(varianceLAD)
    
    x = np.linspace(np.asscalar(Ebeta[0,i] - 3 * sigmaLAD[0,i]), np.asscalar(Ebeta[0,i] + 3 * sigmaLAD[0,i]),100)
    plt.plot(x, mlab.normpdf(x,np.asscalar(Ebeta[0,i]), np.asscalar(sigmaLAD[0,i])))
    i = i + 1
plt.show()
   # beta is Ebeta, 
   
#Shrinkage estimator
i = 0
ssquareSE = np.zeros([1,5])
EbetaSE = np.zeros([1,5])
varianceSE = np.zeros([1,5])
wSE = np.zeros([1,5])
while i < len(bi):
    ssquareSE[0,i] = (1/(n - 2)) * (np.square(excess_return[i,:] - betaJS[0,i] - np.multiply(betaJS[1,i],market_excess_return[:,1])).sum())
    ssquareSE[0,i] = np.divide(ssquareSE[0,i], v[0,i])
    ssquareSEsigma = np.sqrt(ssquareSE)
    # draw of original LAD
    xSE = np.linspace(np.asscalar(betaJS[0,i] - 3 * ssquareSEsigma[0,i]), np.asscalar(betaJS[0,i] + 3 * ssquareSEsigma[0,i]), 100)
    plt.plot(xSE,mlab.normpdf(xSE,np.asscalar(betaJS[0,i]), np.asscalar(ssquareSEsigma[0,i])),color = 'k')
    wSE[0,i] = 1/ssquareSE[0,i] * (1/(1/ssquareSE[0,i] + 1/sigmasquare[0,i]))
    EbetaSE[0,i] = (1-wSE[0,i])*bi[i] + wSE[0,i]*betaJS[0,i]
    varianceSE[0,i] = 1/(1/ssquareSE[0,i] + 1/sigmasquare[0,i])
    sigmaSE = np.sqrt(varianceSE)
    
    x = np.linspace(np.asscalar(EbetaSE[0,i] - 3 * sigmaSE[0,i]), np.asscalar(EbetaSE[0,i] + 3 * sigmaSE[0,i]),100)
    plt.plot(x, mlab.normpdf(x,np.asscalar(EbetaSE[0,i]), np.asscalar(sigmaSE[0,i])))
    i = i + 1
plt.show()

# get the ssquare for alpha sigmasquare
# for IBM
t = np.zeros([1,5])
i = 0
while i < len(bi):
    sigmasquare[0,i] = (1/(n-2)) * (np.square(excess_return[:,i] - ai[i] - np.multiply(bi[i],market_excess_return[:,1])).sum())
    sigmasquare[0,i] = np.sqrt(sigmasquare[0,i])
    t[0,i] = (ai[i]-1)/(sigmasquare[0,i]/np.sqrt(n))
    i = i + 1
t
#use 95% and t-score
# t = 1.984(df = 100) t = 1.962(df = 1000)
#arima rolling forcasting one step by one step 
import scipy
from statsmodels.tsa.arima_model import ARIMA
i = 1
excess_return_forecast = np.zeros([1,5])
market_excess_return_forecast = np.zeros([1,2])


while i <= 60:
    j = 0
    while j < 5:
        
        model_excess = ARIMA(excess_return[:,j], order = (2,0,1))
        model_fit = model_excess.fit()
        excess_return_forecast[0,j] = model_fit.forecast()[0]
        j = j + 1
    excess_return = np.concatenate((excess_return,excess_return_forecast), axis =0)
    model_market = ARIMA(market_excess_return[:,1], order = (2,0,1))
    model_fit_market = model_market.fit()
    market_excess_return_forecast[0,1]=model_fit_market.forecast()[0]
    market_excess_return_forecast[0,0] = 1 # add a column of "1" in to forcasted market return
    market_excess_return = np.concatenate((market_excess_return,market_excess_return_forecast),axis = 0)
    i = i + 1
excess_return.shape
market_excess_return.shape

#since ARIMA meet problem, and i don't know how to fix "LinAlgError: SVD did not converge"
# i will make market_excess_return the last 4 data the same as other data
#a = np.array(([1,0.006770609694241901],[1,0.006770609694241901],[1,0.006770609694241901],[1,0.006770609694241901],[1,0.006770609694241901]))
#b = market_excess_return
#market_excess_return_edited = np.concatenate((a,b), axis = 0)


#[1-131] is original data [132-192] is forcasted
#apply ai, bi to data
from sklearn.metrics import mean_squared_error
from math import sqrt
rmsefix = np.zeros([60,5])
predicted = np.zeros([60,5])
i = 72
while i < 132:
    predicted=market_excess_return[i:i+60,:]*np.array([ai.T,bi.T])
    j = 0
    while j < 5:
        rmsefix[i-72,j] = sqrt(mean_squared_error(excess_return[i:i+60,j], predicted[:,j]))
        j = j + 1
    i = i + 1
rmsefixmean = np.mean(rmsefix,axis = 0)
#cumulative
i = 72
rmseacumu = np.zeros([60,5])
while i < 132:
    predictedAcumu = np.zeros([i+60-72,5])
    predictedAcumu = market_excess_return[72:i+60,:]*np.array([ai.T,bi.T])
    j = 0
    while j < 5:
        rmseacumu[i - 72,j] = sqrt(mean_squared_error(excess_return[72:i+60,j],predictedAcumu[:,j]))
        j = j + 1
    i = i + 1
rmseacumumean = np.mean(rmseacumu,axis = 0)
rmseacumumean
# 5c.
#assume expected return equal to zero, which means excess return = 1
excess_return_expected = np.ones([60,5])
excess_return_original = excess_return[0:132]
excess_return_new = np.concatenate((excess_return_original,excess_return_expected),axis = 0)
# concatenate the expect return = 1 call "new"
#fix rolling estimate
i = 72
rmse_new = np.zeros([60,5])
rmse_new_mean = np.zeros([1,5])
predicted_new = np.zeros([60,5])
while i < 132:
    predicted_new=market_excess_return[i:i+60,:]*np.array([ai.T,bi.T])
    j = 0
    while j < 5:
        rmse_new[i-72,j] = sqrt(mean_squared_error(excess_return[i:i+60,j], predicted_new[:,j]))
        j = j + 1
    i = i + 1
rmse_new_mean = np.mean(rmse_new,axis = 0)
#5d

#1
predict_plot = np.zeros([191,5])

####e ii
[Eai, Ebi] =((market_excess_return[132:,:].T)*market_excess_return[132:,:])**(-1)*(market_excess_return[132:,:].T)*(excess_return[132:,:])#coefficient for forecast value
predict_plot = market_excess_return*np.array([Eai.T,Ebi.T])
plt.plot(market_excess_return[:,1],predict_plot)
plt.show()

#Use the rolling factor model results to allocate your stocks into the maximum Sharpe ratio portfolio.
# to get the weight
M = excess_return.shape[0]#number of rows in excess return
N = excess_return.shape[1]
u1 = np.matrix(np.ones(M)).T
mu0 = excess_return.T*u1/M
sigma0 = (excess_return - mu0.T).T*(excess_return - mu0.T)/(M-1) #convariance matrix
un = np.matrix(np.ones(N)).T
A = un.T*sigma0**(-1)*un #coefficient A in characteristic function
B=un.T*sigma0**(-1)*mu0 #coefficient B in characteristic function
C=mu0.T*sigma0**(-1)*mu0 #coefficient C in characteristic function
G=A*C-B**2 #coefficient G in characteristic function
msrw0=(sigma0**(-1)*mu0)/(un.T*sigma0**(-1)*mu0) #mas Sharpe ratio portfolio weights
#variance is sigma0
var = np.array([0.00605505,0.00431757,0.00220341,0.00142263,0.00305105])
Expected_return = np.mean(predict_plot,axis = 0)
utility = Expected_return - 2*var

# for acumulative 
i = 72
predictedai = np.zeros([60,5])
predictedbi = np.zeros([60,5])
while i < 132:
    [predictedai[i-72,:],predictedbi[i-72,:]] = ((market_excess_return[72:i+60,:].T)*market_excess_return[72:i+60,:])**(-1)*(market_excess_return[72:i+60,:].T)*(excess_return[72:i+60,:])#coefficient for forecast value
    i = i + 1
predictedaimean=np.mean(predictedai,axis=0)
predictedbimean=np.mean(predictedbi,axis=0)
predictreturn = market_excess_return[132:,:]*np.array([predictedaimean.T,predictedbimean.T])
Expected_return_acumu = np.mean(predictreturn,axis = 0)
utility = Expected_return_acumu - 2*var

plt.plot(market_excess_return[132:,1],predictreturn) 
plt.show()

M = predictreturn.shape[0]#number of rows in excess return
N = predictreturn.shape[1]
u1 = np.matrix(np.ones(M)).T
mu0 = predictreturn.T*u1/M
sigma0 = (predictreturn - mu0.T).T*(predictreturn - mu0.T)/(M-1) #convariance matrix
un = np.matrix(np.ones(N)).T
A = un.T*sigma0**(-1)*un #coefficient A in characteristic function
B=un.T*sigma0**(-1)*mu0 #coefficient B in characteristic function
C=mu0.T*sigma0**(-1)*mu0 #coefficient C in characteristic function
G=A*C-B**2 #coefficient G in characteristic function
msrw0=(sigma0**(-1)*mu0)/(un.T*sigma0**(-1)*mu0) #mas Sharpe ratio portfolio w

# plot of expected return and time varying beta


#The Kalman Filter
