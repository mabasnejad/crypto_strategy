#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
#import yfinance as yf
import matplotlib.pyplot as plt
import mlfinlab  as ml
from datetime import datetime,date
from kucoin.client import Client
from kucoin.exceptions import KucoinAPIException
from datetime import datetime,date,timedelta
import pytz
import dateparser
import schedule
import time
from tsmoothie import smoother
#from pyti import money_flow_index
#import investpy
import urllib.request
import json
import requests


# In[7]:


def date_to_seconds(date_str):
    """Convert UTC date to seconds

    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"

    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/

    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    :type date_str: str
    """
    # get epoch value in UTC
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = dateparser.parse(date_str)
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds())


# In[8]:


def get_historical_klines_tv(symbol, interval, start_str, end_str=None):
    """Get Historical Klines from Kucoin (Trading View)

    See dateparse docs for valid start and end string formats http://dateparser.readthedocs.io/en/latest/

    If using offset strings for dates add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"

    :param symbol: Name of symbol pair e.g BNBBTC
    :type symbol: str
    :param interval: Trading View Kline interval
    :type interval: str
    :param start_str: Start date string in UTC format
    :type start_str: str
    :param end_str: optional - end date string in UTC format
    :type end_str: str

    :return: list of OHLCV values

    """

    # init our array for klines
    klines = []
    client = Client("", "","")

    # convert our date strings to seconds
    start_ts = date_to_seconds(start_str)

    # if an end time was not passed we need to use now
    if end_str is None:
        end_str = 'now UTC'
    end_ts = date_to_seconds(end_str)

    kline_res = client.get_kline_data(symbol, interval, start_ts, end_ts)
    if  len(kline_res):
        kline = pd.DataFrame (kline_res, columns = ['time', 'open','close','high','low','ta','volume'])
        kline['time']=pd.to_datetime(kline['time'], unit='s')
        kline=kline.reindex(index=kline.index[::-1])
        kline=kline.set_index('time')
        kline['close']=pd.to_numeric(kline['close'], downcast="float")
        kline['open']=pd.to_numeric(kline['open'], downcast="float")
        kline['high']=pd.to_numeric(kline['high'], downcast="float")
        kline['low']=pd.to_numeric(kline['low'], downcast="float")
        kline['volume']=pd.to_numeric(kline['volume'], downcast="float")

    #print(kline_res[0][0])
    #print(kline)
   # check if we got a result
    
            
    return kline


# In[9]:


def ml_triple_barrier(Data_P,period_D,pt_sl,alpha):
    sp_close=Data_P.close[-2500:]
    sp_close.index=pd.to_datetime(sp_close.index[0:])
    #print(sp_close[-10:])

    daily_vol = ml.util.get_daily_vol(close=sp_close, lookback=int(period_D))
    #print(daily_vol)

    # Apply Symmetric CUSUM Filter and get timestamps for events
    # Note: Only the CUSUM filter needs a point estimate for volatility
    cusum_events = ml.filters.cusum_filter(sp_close,
                                       threshold=daily_vol.mean())
    #print(cusum_events)
    # Compute vertical barrier
    vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events,
                                                     close=sp_close,
                                                     num_days=int(period_D/2))
    #print((vertical_barriers))
    pt_sl = pt_sl     #[2, 2.618]
    #print(daily_vol.mean())
    min_ret = daily_vol.mean()/alpha
    triple_barrier_events = ml.labeling.get_events(close=sp_close,
                                               t_events=cusum_events,
                                               pt_sl=pt_sl,
                                               target=daily_vol,
                                               min_ret=min_ret,
                                               num_threads=1,
                                               vertical_barrier_times=vertical_barriers,
                                               )

    #print(triple_barrier_events[-20:])
    meta_labels = ml.labeling.get_bins(triple_barrier_events, sp_close)
    meta_labels[['t1']]=triple_barrier_events.t1
    meta_labels.insert(4, "Exit_Price", 0, True)
    h_exit=np.array(Data_P['close'].loc[meta_labels['t1']])
    meta_labels["Exit_Price"].iloc[0:]=h_exit
    #print(meta_labels[-5:])
    return(triple_barrier_events,meta_labels)


# In[10]:


def EVB(price,period):
    ################################
    #sp_close_D=pd.Series()
    sp_close_D=price.close
    sp_close_D.index=pd.to_datetime(sp_close_D.index[0:])
    #print(sp_close_D[-10:])
    sp_close_BD=np.log10(sp_close_D)
    #print(sp_close_BD)
    ############################3####
    data1=np.array(sp_close_BD)
    #print(data1[-10:])
    alpha1=(1-np.sin(2*np.pi/period))/np.cos(2*np.pi/period)
    a1=np.exp(-1.414*np.pi/10)
    b1=2*a1*np.cos(1.414*np.pi/10)
    c2=b1
    c3=-a1*a1
    c1=1-c2-c3
    hp=np.zeros((data1.shape[0],1))
    filt=np.zeros((data1.shape[0],1))
    wave=0
    pwr=0
    signal=np.zeros((data1.shape[0],1))
    for i in range(2,data1.shape[0]):
        hp[i]=0.5*(1+alpha1)*(data1[i]-data1[i-1])+alpha1*hp[i-1]
        filt[i]=c1*0.5*(hp[i]+hp[i-1])+c2*filt[i-1]+c3*filt[i-2]
        wave=(filt[i]+filt[i-1]+filt[i-2])/3
        pwr=np.sqrt((filt[i]*filt[i]+filt[i-1]*filt[i-1]+filt[i-2]*filt[i-2])/3)
        signal[i]=wave/pwr
    #print(signal[-5:])
    signal=np.nan_to_num(signal)
    price.loc[:,['signal']]=signal
    
    smoother1 = smoother.KalmanSmoother(component='level_trend',component_noise={'level':1/period, 'trend':1/period})
    #ExponentialSmoother(window_len=period, alpha=0.3)
    #BinnerSmoother(n_knots=int(period))
    #SpectralSmoother(smooth_fraction=0.1, pad_len=period)
    #LowessSmoother(smooth_fraction=1/period, iterations=5)
    smoother1.smooth(signal)
    signal_SM=smoother1.smooth_data.T
    #print(signal_SM.shape)
    #print(signal_SM.shape)
    price.loc[:,['signal_SM']]=signal_SM
    #price['signal_SM']=savgol_filter(price['signal'], int(period_s), order)
    price['signal_yestrday']=price['signal_SM'].shift(1)
    #print(price[-5:])
    return price


# In[11]:


def BPF(price,period):
    ################################
    sp_close_D=price.close
    sp_close_D.index=pd.to_datetime(sp_close_D.index[0:])
    #print(sp_close_D[-10:])
    sp_close_BD=np.log10(sp_close_D)
    #print(sp_close_BD)
    data1=sp_close_BD
    ############################3####
    bandwidth=0.33;
    alpha1= (np.cos(0.25*2*np.pi*bandwidth/period)+np.sin(0.25*2*np.pi*bandwidth/period)-1)/np.cos(0.25*2*np.pi*bandwidth/period)
    hp1=np.zeros((data1.shape[0],1))
    BP=np.zeros((data1.shape[0],1))
    peak=np.zeros((data1.shape[0],1))
    signal_BP=np.zeros((data1.shape[0],1))
    beta1=np.cos(2*np.pi/period)
    gama1=1/np.cos(2*np.pi*bandwidth/period)
    alpha2=gama1-np.sqrt(gama1**2-1)

    for i in range(2,data1.shape[0]):
        hp1[i]=(1-alpha1/2)*(data1[i]-data1[i-1])+(1-alpha1)*hp1[i-1]
    for i in range(2,data1.shape[0]):
        BP[i]=0.5*(1-alpha2/2)*(hp1[i]-hp1[i-2])+beta1*(1+alpha2)*BP[i-1]-alpha2*BP[i-2]
    for i in range(2,data1.shape[0]):
        peak[i]=0.991*peak[i-1];
        if np.abs(BP[i])>peak[i]:
            peak[i]=np.abs(BP[i])
        signal_BP[i]=BP[i]/peak[i]
        #print(signal)
    signal_BP=np.nan_to_num(signal_BP)
    
    price['signal_BP']=signal_BP
    price['BP']=BP
    if period>len(sp_close_D):
        period=len(sp_close_D)
    smoother1 = smoother.BinnerSmoother(n_knots=int(period))
    #KalmanSmoother(component='level_trend',component_noise={'level':1/period, 'trend':1/period})
    #ExponentialSmoother(window_len=period, alpha=0.3)
    #BinnerSmoother(n_knots=int(period))
    #SpectralSmoother(smooth_fraction=0.1, pad_len=period)
    #LowessSmoother(smooth_fraction=1/period, iterations=5)
    smoother1.smooth(signal_BP)
    signal_SM=smoother1.smooth_data.T
    #print(signal_SM.shape)
    #print(signal_SM.shape)
    price['signal_SM']=signal_SM
    #price['signal_SM']=savgol_filter(price['signal'], int(period_s), order)
    price['signal_yestrday']=price['signal_SM'].shift(1)
    #print(price[-10:])
    return price


# In[12]:


def change_time(time_d):
    time1=time_d.isocalendar()
    #print(time1)
    
    if time1[1]==1 and time1[2]<4 :
        time1=(time1[0]-1,52,4)
        time2=datetime.fromisocalendar(time1[0],time1[1],time1[2])
        diff_53=time_d-time2
        if diff_53.days>6:
            time1=(time1[0],53,4)
            #print(time1)
    if time1[2]<4:
        time1=(time1[0],time1[1]-1,4)
    if time1[2]>3:
        time1=(time1[0],time1[1],4)
    
    time2=datetime.fromisocalendar(time1[0],time1[1],time1[2])
    
    #print(time2)
    return time2


# In[13]:


def trade_signal(triple_barrier_events,Data_D,Data_W,period_D2,period_D3,period_W1,period_W2,pt_sl):
    is_NaN = triple_barrier_events.isnull()
    row_has_NaN = is_NaN. any(axis=1)
    Trade_signal= triple_barrier_events[row_has_NaN].copy()
    #print(Trade_signal)
    Trade_signal.insert(4, "action", 0, True)
    Trade_signal.insert(5, "Price", 0, True)
    Trade_signal.insert(6, "TP", 0, True)
    Trade_signal.insert(7, "SL", 0, True)
    Trade_signal.insert(8, "ORDER_TR", 0, True)
    Trade_signal.insert(9, "ORDER_SL", 0, True)
    Trade_signal.insert(10, "ORDER_TP", 0, True)
    Trade_signal.insert(11, "TR-FLAG", False, True)

    for index1,row  in Trade_signal.iterrows():
        #print(index1)
        index_d=change_time(index1)
        dR_D2=pd.DataFrame()
        dR_D3=pd.DataFrame()
        dR_D3_BPF=pd.DataFrame()
        dR_W1=pd.DataFrame()
        dR_W2=pd.DataFrame()
        dR_D2=Data_D.loc[:index1].copy()
        dR_D2=EVB(dR_D2,int(period_D2))
        dR_D3=Data_D.loc[:index1].copy()
        dR_D3=EVB(dR_D3,int(period_D3))
        dR_D_BPF=Data_D.loc[:index1].copy()
        dR_D_BPF=BPF(dR_D_BPF,int(period_D3))
        dR_W1=Data_W.loc[:index_d].copy()
        dR_W1.loc[index_d,['close']]=dR_D2.loc[index1,['close']]
        dR_W1=EVB(dR_W1,int(period_W1))
        dR_W2=Data_W.loc[:index_d].copy()
        dR_W2.loc[index_d,['close']]=dR_D2.loc[index1,['close']]
        dR_W2=EVB(dR_W2,int(period_W2))
        if dR_W1['signal_SM'].loc[index_d]>=0.9 and dR_W2['signal_SM'].loc[index_d]>=0.9:
            if dR_D2['signal_SM'].loc[index1]>= 0.9 and dR_D3['signal_SM'].loc[index1]>= 0.9 and dR_D_BPF['signal_SM'].loc[index1]>= 0.9  : 
                Trade_signal['action'].loc[index1]=1
            if dR_D2['signal_SM'].loc[index1]>= 0.9 and dR_D3['signal_SM'].loc[index1]>dR_D3['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= 0.9 :
                Trade_signal['action'].loc[index1]=1
            if dR_D3['signal_SM'].loc[index1]>= 0.9 and dR_D2['signal_SM'].loc[index1]>dR_D2['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= 0.9 :
                Trade_signal['action'].loc[index1]=1
            if dR_D2['signal_SM'].loc[index1]>dR_D2['signal_yestrday'].loc[index1]  and dR_D3['signal_SM'].loc[index1]>dR_D3['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= 0.9:
                Trade_signal['action'].loc[index1]=1
                  
            if dR_D2['signal_SM'].loc[index1]>= 0.9 and dR_D3['signal_SM'].loc[index1]>= 0.9 and dR_D_BPF['signal_SM'].loc[index1]>= dR_D_BPF['signal_yestrday'].loc[index1]  : 
                Trade_signal['action'].loc[index1]=1
            if dR_D2['signal_SM'].loc[index1]>= 0.9 and dR_D3['signal_SM'].loc[index1]>dR_D3['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= dR_D_BPF['signal_yestrday'].loc[index1] :
                Trade_signal['action'].loc[index1]=1
            if dR_D3['signal_SM'].loc[index1]>= 0.9 and dR_D2['signal_SM'].loc[index1]>dR_D2['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= dR_D_BPF['signal_yestrday'].loc[index1] :
                Trade_signal['action'].loc[index1]=1
            if dR_D2['signal_SM'].loc[index1]>dR_D2['signal_yestrday'].loc[index1]  and dR_D3['signal_SM'].loc[index1]>dR_D3['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= dR_D_BPF['signal_yestrday'].loc[index1]:
                Trade_signal['action'].loc[index1]=1
        if dR_W1['signal_SM'].loc[index_d]>=0.9 and dR_W2['signal_SM'].loc[index_d]>dR_W2['signal_yestrday'].loc[index_d] :
            if dR_D2['signal_SM'].loc[index1]>= 0.9 and dR_D3['signal_SM'].loc[index1]>= 0.9 and dR_D_BPF['signal_SM'].loc[index1]>= 0.9  : 
                Trade_signal['action'].loc[index1]=1
            if dR_D2['signal_SM'].loc[index1]>= 0.9 and dR_D3['signal_SM'].loc[index1]>dR_D3['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= 0.9 :
                Trade_signal['action'].loc[index1]=1
            if dR_D3['signal_SM'].loc[index1]>= 0.9 and dR_D2['signal_SM'].loc[index1]>dR_D2['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= 0.9 :
                Trade_signal['action'].loc[index1]=1
            if dR_D2['signal_SM'].loc[index1]>dR_D2['signal_yestrday'].loc[index1]  and dR_D3['signal_SM'].loc[index1]>dR_D3['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= 0.9:
                Trade_signal['action'].loc[index1]=1
                  
            if dR_D2['signal_SM'].loc[index1]>= 0.9 and dR_D3['signal_SM'].loc[index1]>= 0.9 and dR_D_BPF['signal_SM'].loc[index1]>= dR_D_BPF['signal_yestrday'].loc[index1]  : 
                Trade_signal['action'].loc[index1]=1
            if dR_D2['signal_SM'].loc[index1]>= 0.9 and dR_D3['signal_SM'].loc[index1]>dR_D3['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= dR_D_BPF['signal_yestrday'].loc[index1] :
                Trade_signal['action'].loc[index1]=1
            if dR_D3['signal_SM'].loc[index1]>= 0.9 and dR_D2['signal_SM'].loc[index1]>dR_D2['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= dR_D_BPF['signal_yestrday'].loc[index1] :
                Trade_signal['action'].loc[index1]=1
            if dR_D2['signal_SM'].loc[index1]>dR_D2['signal_yestrday'].loc[index1]  and dR_D3['signal_SM'].loc[index1]>dR_D3['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= dR_D_BPF['signal_yestrday'].loc[index1]:
                Trade_signal['action'].loc[index1]=1
            
        if dR_W1['signal_SM'].loc[index_d]>dR_W1['signal_yestrday'].loc[index_d]  and dR_W2['signal_SM'].loc[index_d]>=0.9 :
            if dR_D2['signal_SM'].loc[index1]>= 0.9 and dR_D3['signal_SM'].loc[index1]>= 0.9 and dR_D_BPF['signal_SM'].loc[index1]>= 0.9  : 
                Trade_signal['action'].loc[index1]=1
            if dR_D2['signal_SM'].loc[index1]>= 0.9 and dR_D3['signal_SM'].loc[index1]>dR_D3['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= 0.9 :
                Trade_signal['action'].loc[index1]=1
            if dR_D3['signal_SM'].loc[index1]>= 0.9 and dR_D2['signal_SM'].loc[index1]>dR_D2['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= 0.9 :
                Trade_signal['action'].loc[index1]=1
            if dR_D2['signal_SM'].loc[index1]>dR_D2['signal_yestrday'].loc[index1]  and dR_D3['signal_SM'].loc[index1]>dR_D3['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= 0.9:
                Trade_signal['action'].loc[index1]=1
                  
            if dR_D2['signal_SM'].loc[index1]>= 0.9 and dR_D3['signal_SM'].loc[index1]>= 0.9 and dR_D_BPF['signal_SM'].loc[index1]>= dR_D_BPF['signal_yestrday'].loc[index1]  : 
                Trade_signal['action'].loc[index1]=1
            if dR_D2['signal_SM'].loc[index1]>= 0.9 and dR_D3['signal_SM'].loc[index1]>dR_D3['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= dR_D_BPF['signal_yestrday'].loc[index1] :
                Trade_signal['action'].loc[index1]=1
            if dR_D3['signal_SM'].loc[index1]>= 0.9 and dR_D2['signal_SM'].loc[index1]>dR_D2['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= dR_D_BPF['signal_yestrday'].loc[index1] :
                Trade_signal['action'].loc[index1]=1
            if dR_D2['signal_SM'].loc[index1]>dR_D2['signal_yestrday'].loc[index1]  and dR_D3['signal_SM'].loc[index1]>dR_D3['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= dR_D_BPF['signal_yestrday'].loc[index1]:
                Trade_signal['action'].loc[index1]=1
        
        if dR_W1['signal_SM'].loc[index_d]>dR_W1['signal_yestrday'].loc[index_d]  and dR_W2['signal_SM'].loc[index_d]>dR_W2['signal_yestrday'].loc[index_d] :
            if dR_D2['signal_SM'].loc[index1]>= 0.9 and dR_D3['signal_SM'].loc[index1]>= 0.9 and dR_D_BPF['signal_SM'].loc[index1]>= 0.9  : 
                Trade_signal['action'].loc[index1]=1
            if dR_D2['signal_SM'].loc[index1]>= 0.9 and dR_D3['signal_SM'].loc[index1]>dR_D3['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= 0.9 :
                Trade_signal['action'].loc[index1]=1
            if dR_D3['signal_SM'].loc[index1]>= 0.9 and dR_D2['signal_SM'].loc[index1]>dR_D2['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= 0.9 :
                Trade_signal['action'].loc[index1]=1
            if dR_D2['signal_SM'].loc[index1]>dR_D2['signal_yestrday'].loc[index1]  and dR_D3['signal_SM'].loc[index1]>dR_D3['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= 0.9:
                Trade_signal['action'].loc[index1]=1
                  
            if dR_D2['signal_SM'].loc[index1]>= 0.9 and dR_D3['signal_SM'].loc[index1]>= 0.9 and dR_D_BPF['signal_SM'].loc[index1]>= dR_D_BPF['signal_yestrday'].loc[index1]  : 
                Trade_signal['action'].loc[index1]=1
            if dR_D2['signal_SM'].loc[index1]>= 0.9 and dR_D3['signal_SM'].loc[index1]>dR_D3['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= dR_D_BPF['signal_yestrday'].loc[index1] :
                Trade_signal['action'].loc[index1]=1
            if dR_D3['signal_SM'].loc[index1]>= 0.9 and dR_D2['signal_SM'].loc[index1]>dR_D2['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= dR_D_BPF['signal_yestrday'].loc[index1] :
                Trade_signal['action'].loc[index1]=1
            if dR_D2['signal_SM'].loc[index1]>dR_D2['signal_yestrday'].loc[index1]  and dR_D3['signal_SM'].loc[index1]>dR_D3['signal_yestrday'].loc[index1] and dR_D_BPF['signal_SM'].loc[index1]>= dR_D_BPF['signal_yestrday'].loc[index1]:
                Trade_signal['action'].loc[index1]=1
    
    for index2,row1  in Trade_signal.iterrows():
        if row1.action!=0:
            Trade_signal.loc[index2,['Price']]=Data_D['close'].loc[index2]
             #print("Price="+str(d1['close'].loc[index2]))
            #print(pt_sl[0])
            Trade_signal.loc[index2,['TP']]=(Data_D['close'].loc[index2]+Data_D['close'].loc[index2]*pt_sl[1]*row1.trgt*row1.action)
             #print("TP="+str(round(TP1)))
            Trade_signal.loc[index2,['SL']]=(Data_D['close'].loc[index2]-Data_D['close'].loc[index2]*pt_sl[0]*row1.trgt*row1.action)
    return(Trade_signal)


# In[14]:


def check_order(Coin_orders,P_daily,Meta_label,api_key,api_secret,api_passphrase):
    coin_order_n=Coin_orders[Coin_orders['TR-FLAG']==False]
    #print(coin_order_n)
    
    p_close=(P_daily['open'].iloc[-1:])
    p_high=(P_daily['high'].iloc[-1:])
    p_low=(P_daily['low'].iloc[-1:])
    if not(coin_order_n.empty):
        client=Client(api_key,api_secret,api_passphrase)
        for index1,row  in coin_order_n.iterrows():
            if p_low[0]<row['SL']:
                order_id=str(row['ORDER_TP'])
                Coin_orders['TR-FLAG'].loc[index1]=True
                print(order_id)
                try:
                    res = client.cancel_order(order_id)
                    print(res)
                
                except KucoinAPIException as e:
                    print(e.status_code)
                    print(e.message)
                    pass
            if p_high[0]>row['TP']:
                order_id=str(row['ORDER_SL'])
                Coin_orders['TR-FLAG'].loc[index1]=True
                print(order_id)
                try:
                    res = client.cancel_order(order_id)
                    print(res)
                
                except KucoinAPIException as e:
                    print(e.status_code)
                    print(e.message)
                    pass
            ac = Meta_label[Meta_label.index==index1]
            if not(ac.empty) and not(coin_order_n['TR-FLAG'].loc[index1]):
                Coin_orders['TR-FLAG'].loc[index1]=True
                #print(1000)
                try:
                    order_get = client.get_order(row.ORDER_TR)
                    print(order_get)
                    order_tr = client.create_market_order(symbol=order_get['symbol'], side=Client.SIDE_SELL, size=order_get['size'])
                    res_TP = client.cancel_order(row.ORDER_TP)
                    res_SL = client.cancel_order(row.ORDER_SL)
                except KucoinAPIException as e:
                    print(e.status_code)
                    print(e.message)
                    pass
            
            
    return(Coin_orders)


# In[15]:


def ip_country():
    #with urllib.request.urlopen("http://ip-api.com/json/") as url:
    with urllib.request.urlopen("http://ipinfo.io/json/") as url:
        data = json.loads(url.read().decode())
    #url='http://ip-api.com/json/'+ data['IPv4']
    #print(url)
    #response=requests.get(url).json()
    print(data['country'])
    return(data['country'])


# In[18]:


def job(symbol,period_D1,period_D2,period_D3,period_W1,period_W2,capital,vpn,pt_sl,api_key,api_secret,api_passphrase):
    today=datetime.utcnow().date()#- timedelta(4)
    print(datetime.now())
    print(today)
    #INPUT
    country=ip_country()
    if country==vpn:
        ###########################################################################Input
        Coin_ORDER=pd.DataFrame()
        FILE_NAME='D:\\'+symbol+'.xlsx'
        Coin_ORDER=pd.read_excel(FILE_NAME,engine='openpyxl',sheet_name='Sheet1')
        Coin_ORDER.set_index('date',inplace=True)
        print(Coin_ORDER)
        period_D1=period_D1
        period_D2=period_D2
        period_D3=period_D3
        period_W1=period_W1
        period_W2=period_W2
        pt_sl=[2, 2.618]
        alpha=1
        capital=capital
        start_time="1 Nov, 2016"
        symbol=symbol
        api_key=api_key
        api_secret=api_secret
        api_passphrase=api_passphrase
        #########################################################
        try:
            time_frames="1week"
            d_week= get_historical_klines_tv(symbol, time_frames, start_time)
            time_frames="1day"
            d_Daily= get_historical_klines_tv(symbol, time_frames, start_time)
            print("Daily Price")
            print(d_Daily[-5:])
        except KucoinAPIException as e:
            print(e.status_code)
            print(e.message)
            if e.status_code==429:
                time.sleep(100)
                time_frames="1week"
                d_week= get_historical_klines_tv(symbol, time_frames, start_time)
                time_frames="1day"
                d_Daily= get_historical_klines_tv(symbol, time_frames, start_time)
                print("Daily Price")
                print(d_Daily[-5:])
        import mlfinlab  as ml
        ##########################################################
        triple_barrier_events,Meta_label_=ml_triple_barrier(Data_P=d_Daily,period_D=period_D1,pt_sl=pt_sl,alpha=alpha)
        print("Meta_labels")
        print(Meta_label_[-10:])
        is_NaN = triple_barrier_events.isnull()
        row_has_NaN = is_NaN. any(axis=1)
        triple_barrier_events_NA= triple_barrier_events[row_has_NaN].copy()
        trade_signal_=trade_signal(triple_barrier_events=triple_barrier_events_NA,Data_D=d_Daily,Data_W=d_week,period_D2=period_D2,period_D3=period_D3,period_W1=period_W1,period_W2=period_W1,pt_sl=pt_sl)
        print("Trade_signal")
        print(trade_signal_)
        ###########################################################
        if not(Coin_ORDER.empty):
            Coin_ORDER=check_order(Coin_ORDER,d_Daily,Meta_label_,api_key,api_secret,api_passphrase)
            print("Coin_order")
            print(Coin_ORDER)
        #######################################33333    
        #print(trade_signal_.empty)
        my_time = "23:00:00"
        my_time = datetime.strptime(my_time, "%H:%M:%S").time()
        print(my_time)
        if not(trade_signal_.empty) and datetime.utcnow().time()>my_time:
            print("hh")
            if trade_signal_.index[-1]==today and trade_signal_['action'].iloc[-1]==1 :
                client=Client(api_key,api_secret,api_passphrase)
                print(today)
                tickers = client.get_ticker(symbol)
                n_o_r=tickers['bestBid'][::-1].find('.')
                #print(n_o_r)
                #print(tickers)
                size_tr=str(int(capital/float(tickers['bestBid'])))
                print("size_tr="+size_tr)
                order_tr = client.create_market_order(symbol, Client.SIDE_BUY, size=size_tr)
                print(order_tr['orderId'])
                SL_tr=str(round(trade_signal_['SL'].iloc[-1],n_o_r+1))
                TP_tr=str(round(trade_signal_['TP'].iloc[-1],n_o_r+1))
                print("SL_tr="+SL_tr)
                print("TP_tr"+TP_tr)
                trade_signal_['SL'].iloc[-1]=float(SL_tr)
                trade_signal_['TP'].iloc[-1]=float(TP_tr)
                time.sleep(200)
                order_SL = client.create_limit_order(symbol, Client.SIDE_SELL,SL_tr , size_tr,stop='loss',stop_price=SL_tr)
                order_TP = client.create_limit_order(symbol, Client.SIDE_SELL,TP_tr , size_tr,stop='entry',stop_price=TP_tr)
                trade_signal_['ORDER_TR'].iloc[-1]=order_tr['orderId']
                trade_signal_['ORDER_SL'].iloc[-1]=order_SL['orderId']
                trade_signal_['ORDER_TP'].iloc[-1]=order_TP['orderId']
                trade_signal_.index.name='date'
                Coin_ORDER=Coin_ORDER.append(trade_signal_[-1:],ignore_index=False)
                print((Coin_ORDER))
        
        Coin_ORDER.to_excel(FILE_NAME)
            


# In[ ]:





# In[ ]:




