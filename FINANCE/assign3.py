#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 17:56:44 2019

@author: linqi
"""

#trading algorithm 
# moving avarage 
import time
import fix_yahoo_finance as yf
start = '2007-12-01'
end = '2018-11-30'
tickers = ['AMZN']
data = yf.download(tickers,start,end)

#get the monthly data, get the last day data in a month
pmo = data['Adj Close'].resample('D').last()
import numpy as np
import pandas as pd
import datetime 
import matplotlib.pyplot as plt
#holding period return
pmo = pmo.dropna()
pmo.index = np.arange(1, len(pmo) + 1)
rmo = pmo[1:len(pmo)].values/pmo[0:len(pmo)-1]
rmo.index = np.arange(1, len(rmo) + 1)
rmo = rmo.dropna()

#R['MA20'] = rmo.rolling(20).mean()
#R['MA50'] = rmo.rolling(50).mean() i don't understand why this doesn't work
returnMA = pd.DataFrame({'MA20':rmo.rolling(20).mean(),'MA50':rmo.rolling(50).mean()})
returnMA.index = np.arange(1, len(returnMA) + 1)
returnMA = returnMA.iloc[50:,:]
pmo = pmo[51:]
rmo = rmo[50:]
pmo.index = np.arange(1, len(pmo) + 1)
rmo.index = np.arange(1, len(rmo) + 1)
returnMA['differenc'] = returnMA.MA20-returnMA.MA50

#import os
#returnMA.to_csv(os.path.expanduser('~/Documents/2019 Spring/QFA/MA.csv'))
#returnMA.to_csv()
#pmo.to_csv(os.path.expanduser('~/Documents/2019 Spring/QFA/MA.csv'))       
returnMA.index = np.arange(1, len(pmo) + 1)
asset = np.zeros((len(returnMA),1))#the number of shares i have
value = np.zeros((len(returnMA),1))# number of shares * price=cost
income = np.zeros((len(returnMA),1))
profit = np.zeros((len(returnMA),1)) # income - value = profit
# first i am going to built moving average, so when ma20 > ma50, i am going to buy 50
# i need to go throught all the time line i will change rmo and pmo to matrix
Return1 = np.zeros((len(returnMA),1))
AcumuReturn1 = np.zeros((len(returnMA),1))#becasue i use add to acumulate, so initiolize as 0
MonthReturn1 = np.ones((len(returnMA),1))

DailyPrice = pmo.to_numpy()
MoveAverage= returnMA.to_numpy()
i = 0
#count = 0
# i need a variable to count the distance from last asset[i] = 0
count = 0
#if differenc[i]>0 and differenc[i]*difference[i+1] >0:
Starttime = np.zeros((len(returnMA),1))
Endtime = np.zeros((len(returnMA),1))
Totaltime = np.zeros((len(returnMA),1))
timecount = np.zeros((len(returnMA),1))
while i < len(returnMA):
    #if i = 2718, which means i + 1 = 2719
    
    Starttime[i]=time.time()
    if i != 2718 and MoveAverage[i,2] > 0 and MoveAverage[i,2]*MoveAverage[i+1,2] >0:
    #count = count + 1
        count = count + 1
        timecount[i] = 1
    # asset = count * 50
        asset[i] = count * 50
    #if count == 1
        if count == 1:
            
        # then value[i] = price[i] * asset[i]
            value[i] = DailyPrice[i] * asset[i]    
        # if count != 1
        else:
            value[i] = value[i - 1] + DailyPrice[i] * 50
       # then value[i] = value[i - 1] + price[i] * 50
#if differenc[i] > 0 and differenc[i]*difference[i+1] <0:
    if i != 2718 and MoveAverage[i,2] > 0 and MoveAverage[i,2]*MoveAverage[i+1,2] <0 and i != 2718:
        
    #count = 0 
        count = 0 
        timecount[i] = -1
    #asset[i] = count * 50=0 
        asset[i] = count * 50
    #value[i] = value[i - 1] + price[i] * asset[i]
        value[i] = value[i - 1] + DailyPrice[i] * asset[i]
    #income[i] = price[i] * asset[i - 1] profit[i] = income[i] - value[i]
        income[i] = DailyPrice[i] * asset[i - 1] 
        profit[i] = income[i] - value[i]
        Return1[i] = profit[i]/value[i] 
    # if differenc[i]<0 and differenc[i]*difference[i+1] >0:
    if i != 2718 and MoveAverage[i,2] < 0 and MoveAverage[i,2]*MoveAverage[i+1,2] >0 and i != 2718:  
        #count += 1
        count = count + 1
        timecount[i] = -1
        #asset[i] = count * (-50)
        asset[i] = count * (-50)
    #if count == 1
        if count == 1:
        # then value[i] = price[i] * asset[i]
            value[i] = DailyPrice[i] * asset[i]
   # if count != 1
        if count != 1:
            value[i] = value[i - 1] + DailyPrice[i] * (-50)
        
       # then value[i] = value[i - 1] + price[i] * (-50)
#if differenc[i]<0 and differenc[i]*difference[i+1] <0:
    if i != 2718 and MoveAverage[i,2] < 0 and MoveAverage[i,2]*MoveAverage[i+1,2] <0 and i != 2718:  
        
    #count = 0
        count = 0
        timecount[i] = 1
    #asset[i] = count * (-50)=0
        asset[i] = count * (-50)
    #value[i] =  value[i - 1] + price[i] * asset[i]
        value[i] =  value[i - 1] + DailyPrice[i] * asset[i]
    #income[i]= price[i] * asset[i - 1]
        income[i]= DailyPrice[i] * asset[i - 1]
    #profit[i] = income[i] - value[i]
        profit[i] = income[i] - value[i]
        Return1[i] = profit[i]/value[i]
# i = i + 1
# if i = 2718
    Endtime[i] =  time.time()
    Totaltime[i] = Endtime[i] - Starttime[i]
    if i == 2718:
        
    # in this case, we need to clean out position
    # value = value[i - 1] 
        value[i] = value[i - 1] 
        
    # income[i] = DailyPrice[i] * asset[i - 1] 
        income[i] = DailyPrice[i] * asset[i - 1] 
    #profit[i] = income[i] - value[i]
        profit[i] = income[i] - value[i]
        Return1[i] = profit[i]/value[i]
    i = i + 1
    
#Totalprofit = profit.sum()
Totalprofit = profit.sum()
#cumulative return
#initionize AcumuReturn[i] = 1
i = 1 
AcumuReturn1[0] = 0
while i < len(returnMA):
    AcumuReturn1[i] = AcumuReturn1[i-1] + Return1[i] 
    i = i + 1
plt.plot(returnMA.index,AcumuReturn1[:,0])
plt.show()

plt.plot(returnMA.index,profit[:,0])
plt.show()
plt.plot(returnMA.index,Return1[:,0])
plt.show()
plt.plot(returnMA.index,Return1[:,0])
plt.show()
#monthly return

#initilize monthreturnplot = []
monthreturnplot1 = np.ones((1,2))
# use daily return(when it is not 0) in a month, Continually multiply, then take 30 sqrt root
#i need a loop every 30 (also loop from i = 1)
i = 1
# while
while i < len(returnMA):
# when i % 30 = 0
    if i %30 == 0:
        
    ## if return != 0
    #monthreturn = return[i] * monthreturn[i - 1] all non-zero return multiply, may be should do add
        MonthReturn1[i] = Return1[i-30:i].sum()
    #monthreturn[i] monthreturn[i]**(1/30) - 1
        #MonthReturn1[i]= MonthReturn1[i]**(1/30) - 1
    #monthreturnplot = numpy.vstack([monthreturn,[i,monthreturn[i]]])
        #a = [[i,MonthReturn1[i]]]
        #monthreturnplot1 = np.vstack([monthreturnplot1,a])
    #initilize MonthReturn1[i] back to 1
        #MonthReturn1[i] = 1
# when i % 30 !=0
    # if return != 0
    #if i % 30 != 0 and Return1[i] != 0:
    #monthreturn = return[i] * monthreturn[i - 1]
       # MonthReturn1[i] = Return1[i] * MonthReturn1[i - 1]
# i = i + 1
    i = i + 1
MonthReturn1=MonthReturn1[np.nonzero(MonthReturn1)]
a=np.linspace(1,len(MonthReturn1),len(MonthReturn1))
#plt.plot(monthreturn[:,0],monthreturn[:,1])
plt.plot(a,MonthReturn1)
plt.show()

# max drawdown
#get the max negative profit
# MDD1 = profit.min()
MDD1 = profit.min()


#max turnover
maxturn = max(asset.min(),asset.max(),key = abs)

#sharpe ratio (mean / std of excess return)
#risk free rate, i need the risk free rate match stock date, no i only need an average 5.69% = 0.0569
#return, average return Return1.mean()
#volitility of the price 


ExReturn = (Return1 - 0.569).mean()
sigma1 = (Return1 - 0.569).std()
sharpe1 = ExReturn/sigma1


#if buy & hold
#sharpehold = (rmo - 0.569).mean()/(rmo - 0.569).std()

# if add commision fee
# assume trading fee per hand $5 
# Totalprofit - 5*2718
Totalprofitafterfee= Totalprofit - 5*2718
#buy and hold
#totalprofit of buy and hold
# DailyPrice[2718] * asset.mean() - DailyPrice[1] * asset.mean() - 10
BHTotalprofitafterfee =DailyPrice[2718] * 100 - DailyPrice[1] *100 - 10



averagetime = Totaltime.mean()
plt.plot(returnMA.index,timecount[:,0])
plt.show()
cumuprofit= np.zeros((len(returnMA),1))
i = 1 
cumuprofit[0] = profit[0]
while i < len(returnMA):
    
    cumuprofit[i] = cumuprofit[i-1]+profit[i]
    i = i + 1
plt.plot()
plt.plot(returnMA.index,cumuprofit[:,0])
plt.show()

#significantly different from zero or not:

# t = Return1/((Return1.std()/sqrt(2719))
t = Return1.mean()/((Return1.std()/np.sqrt(2718))
#h
# when ma20 = ma50, make sure my asset = 0(sell)
# when ma20 < ma50, sell 50
#when  ma20 = 50, make sure asset = 0(buy)
# if after one line, the data immediately change direction?
# so i need to find a price to calculate how much money i make each time
# we can do, if next data, ma20 and ma50 going to change direction, get the average of the two price, as the avearage price to sell/buy

#plt.plot(pmo.index, pmo.values)
#plt.plot(rmo.index, rmo.values)
#plt.show()
#plt.plot(returnMA.index, returnMA.MA20,'r')
#plt.plot(returnMA.index, returnMA.MA50,'b')
#plt.show()
# build the currency arbitrage algorithm:
#i need two area three currencies exchange, e.g. euro to usd, usd to aud, aud to euro
# 1 euro = 1.13 usd 
# 1 usd = 1.40 aud 
# 1 euro = 1.58 aud

#read excel data
import xlrd 
import os
path = os.path.expanduser('~/Documents/2019 Spring/QFA/AUDSDREURO.xlsx')
x1=pd.ExcelFile(path)
AUDEUROSDR=x1.parse('Sheet1')
AUDEUROSDR=AUDEUROSDR.iloc[0:2997,:]

path = os.path.expanduser('~/Documents/2019 Spring/QFA/audeurodollar.xlsx')
x2=pd.ExcelFile(path)
AUDEUROdollar=x2.parse('Sheet1')


AUDEUROSDR = AUDEUROSDR.dropna()
AUDEUROdollar=AUDEUROdollar.dropna()

audeuro = pd.merge(AUDEUROSDR,AUDEUROdollar,left_on = 'Date', right_on='Date')
#to change aud sdr and euro sdr to euro = aud
audeuro['eurotoaud']= 1/audeuro['EUR'] * audeuro['AUD'] 
audeuro.columns = ['Date','AUD','EUR','dollarAUD','dollarEuro','eurotoaud']
#plt.plot(audeuro.Date,audeuro.dollarAUD,color = 'r')
#plt.plot(audeuro.Date,audeuro.dollarEuro,color = 'b')
#plt.show()
audeuro['dollarAUD'] = 1/ audeuro['dollarAUD']
audeuro=audeuro.drop(['Date','AUD','EUR'],axis = 1)
profit = np.zeros((len(audeuro),1))
audeuro = audeuro.to_numpy()
arr = []
arr = audeuro[:,0]*audeuro[:,1]
audeuro1 = np.column_stack([audeuro,arr])
Starttime1 = np.zeros((len(audeuro),1))
Endtime1 = np.zeros((len(audeuro),1))
Totaltime1 = np.zeros((len(audeuro),1))
#i = 0
i = 0

while i < len(profit):
#if audeuro[i,0]-audeuro[i,1]>0
    Starttime1[i]=time.time()
    if audeuro1[i,2]-audeuro1[i,3]>0:
         profit[i] = 1000000*(audeuro1[i,2]-audeuro1[i,3])   
    #profit[i] = 1000000*(audeuro[i,0]-audeuro[i,1])
#if audeuro[i,0]-audeuro[i,1]<0
    if audeuro1[i,2]-audeuro1[i,3]<0:
    
    #profit[i] = 1000000*(-audeuro[i,0]+audeuro[i,1])
        profit[i] = 1000000*(-audeuro1[i,2]+audeuro1[i,3])  
# i = i + 1
    Endtime1[i]=time.time()
    Totaltime1[i] = Endtime1[i] - Starttime1[i]
    i = i + 1
#totalprofitcurrency = profit.sum()
totalprofitcurrency = profit.sum()
averagetime1 = Totaltime1.mean()
AUDEUROdollar = AUDEUROdollar.iloc[:-1,:]
cumuprofit= np.zeros((len(audeuro),1))
i = 1 
cumuprofit[0] = profit[0]
while i < len(audeuro):
    
    cumuprofit[i] = cumuprofit[i-1]+profit[i]
    i = i + 1
plt.plot(AUDEUROdollar.index,cumuprofit[:,0])
plt.show()
# option trading strategy
#delta neutrel strategy
import os
path = os.path.expanduser('~/Documents/2019 Spring/QFA/call.csv')
call = pd.read_csv(path)
call=call.drop(['secid','cp_flag','index_flag','expire date'],axis = 1)
from datetime import timedelta
call['Date'] = call['date'].apply(str)
call['date'] = pd.to_datetime(call.Date)
call = call.drop(['Date'],axis = 1)
call.loc[:,'ExpireDate'] =list(map(lambda x,y: x + timedelta(y),call['date'],call['days']))
call = call[call.date != call.date.shift(1)]
call.index = np.arange(1,len(call)+1)


#clean put
path = os.path.expanduser('~/Documents/2019 Spring/QFA/PUT.csv')
put = pd.read_csv(path)
put=put.drop(['secid','cp_flag','index_flag'],axis = 1)
put['Date'] = put['date'].apply(str)
put['date'] = pd.to_datetime(put.Date)
put = put.drop(['Date'],axis = 1)
put.loc[:,'ExpireDate'] =list(map(lambda x,y: x + timedelta(y),put['date'],put['days']))
put = put[put.date != put.date.shift(1)]
put.index = np.arange(1,len(put)+1)
call.columns = ['date','days','calldelta','callstrike','callpremium','CallExpireDate']
#put.colomns = ['date','days','putdelta','putstrike','putpremium','PutExpireDate']
#why this doesn't work?
put = put.rename(columns={'date':'date','days':'days','delta':'putdelta','impl_strike':'putstrike','impl_premium':'putpremium','ExpireDate':'PutExpireDate'})
Option = pd.merge(call,put,left_on = 'CallExpireDate', right_on='PutExpireDate')

#merge the two
#find IBM spot price

start = '2016-01-04'
end = '2017-12-29'
tickers = ['IBM']
IBMraw = yf.download(tickers,start,end)
IBM = IBMraw['Adj Close'].resample('D').last()
IBM = IBM.dropna()

IBM = pd.DataFrame(index = IBM.index, data=IBM.values)
IBM['Date']=IBM.index
IBM.columns = ['Price','Date']
IBM.index=np.arange(1,len(IBM)+1)
IBM['Price'] = pd.to_numeric(IBM['Price'])
IBM['Date'] = pd.to_datetime(IBM['Date'],errors='coerce')
#IBM['Date'] = list(map(lambda x,y: ))#cannot use this function
#because it is only dealing with one column
#first i need to substring it
#then i need to change it to datetime

path = os.path.expanduser('~/Documents/2019 Spring/QFA/IBMprice.csv')
IBM.to_csv(path)

IBMall = pd.merge(Option,IBM,left_on = 'CallExpireDate',right_on = 'Date')
path = os.path.expanduser('~/Documents/2019 Spring/QFA/optionstretagy.csv')
IBMall.to_csv(path)
IBMmatrix = pd.DataFrame(data = IBMall, columns =['callstrike','callpremium','putstrike','putpremium','Price'])
IBMmatrix1 = IBMmatrix.to_numpy()
#price = IBM['Price'].to_numpy()
#build a new np matrix to collect cost
cost = np.zeros((len(IBMmatrix1),1))
#build a new np matrix to collect income
income = np.zeros((len(IBMmatrix1),1))
#build a new np matrix to collect profit
profit = np.zeros((len(IBMmatrix1),1))
#algorithm
i = 0
Starttime2 = np.zeros((len(IBMmatrix1),1))
Endtime2 = np.zeros((len(IBMmatrix1),1))
Totaltime2 = np.zeros((len(IBMmatrix1),1))
Return3=np.zeros((len(IBMmatrix1),1))
while i < len(IBMall):
#long 4 call20*4 1 put 1*(-80)
#cost 4 * IBMmatrix1[i,1] + 1*IBMmatrix1[i,3] 
    Starttime2[i]=time.time()
    cost[i,0] = 4 * IBMmatrix1[i,1] + 1*IBMmatrix1[i,3]
#income if IBMmatrix1[i,0](call strik) - IBMmatrix1[i,4](price) <0 
    income[i,0] = np.where(IBMmatrix1[i,0]- IBMmatrix1[i,4] <0,IBMmatrix1[i,4]-IBMmatrix1[i,2],0) + np.where(IBMmatrix1[i,2]- IBMmatrix1[i,4] > 0,IBMmatrix1[i,2]- IBMmatrix1[i,4],0)
    #= (IBMmatrix1[i,2](call strik) - IBMmatrix1[i,4](price))+
    #else:
    # =0
    # 
     #if IBMmatrix1[i,0](put strik) - IBMmatrix1[i,4](price))>0
    #IBMmatrix1[i,0](put strik) - IBMmatrix1[i,4](price))
    #else:
    # = 0
#profit = income[i] - cost[i]
    profit[i,0] = income[i,0] - cost[i,0]
    Return3[i] = profit[i,0]/cost[i,0] - 1
    Endtime2[i]=time.time()
    Totaltime2[i] = Endtime2[i] - Starttime2[i]
    i = i + 1
    #totalprofit = profit.sum()
totalprofit = profit.sum()
averagetime2 = Totaltime2.mean()
cumuprofit= np.zeros((len(IBMmatrix1),1))
i = 1 
cumuprofit[0] = profit[0]
while i < len(IBMmatrix1):
    
    cumuprofit[i] = cumuprofit[i-1]+profit[i]
    i = i + 1
plt.plot(IBMmatrix.index,cumuprofit[:,0])
plt.show()
plt.plot(IBMmatrix.index,Return3[:,0])
plt.show()

cumureturn3= np.zeros((len(IBMmatrix1),1))
i = 1
while i < len(IBMmatrix1):
    
    cumureturn3[i] = cumureturn3[i-1]+Return3[i]
    i = i + 1
plt.plot(IBMmatrix.index,cumureturn3[:,0])
plt.show()
sharpe3 = (Return3-0.569).mean()/(Return3-0.569).std()
