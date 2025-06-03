import os
import numpy as np
import math
import scipy.io as sio
from scipy.fftpack import fft,ifft


def DE_PSD(data, fre, time_window):
    '''
    compute DE and PSD
    --------
    input:  data [n*m]          n electrodes, m time points
            stft_para.stftn     frequency domain sampling rate
            stft_para.fStart    start frequency of each frequency band
            stft_para.fEnd      end frequency of each frequency band
            stft_para.window    window length of each sample point(seconds)
            stft_para.fs        original frequency                           采样频率 等于fre
    output: psd,DE [n*l*k]        n electrodes, l windows, k frequency bands
    '''
    #initialize the parameters
    # STFTN=stft_para['stftn']
    # fStart=stft_para['fStart']
    # fEnd=stft_para['fEnd']
    # fs=stft_para['fs']
    # window=stft_para['window']

    #STFTN==短时傅里叶变换 用于频域分析

    STFTN = 200                      #短时傅里叶变换的频域采样率，这里设为 200
    fStart = [1, 4, 8, 14, 31]       #不同频带的起始频率和结束频率
    fEnd = [4, 8, 14, 31, 99]
    window = time_window
    fs = fre

    WindowPoints=fs*window           #每个窗口中的时间点数

    fStartNum=np.zeros([len(fStart)],dtype=int)  #每个频带的起始和结束频率在频域中对应的索引位置
    fEndNum=np.zeros([len(fEnd)],dtype=int)
    for i in range(0,len(fStart)):
        fStartNum[i]=int(fStart[i]/fs*STFTN)     #i频率除采样频率 再乘频率采样率  得到对饮 采样点索引
        fEndNum[i]=int(fEnd[i]/fs*STFTN)

    #print(fStartNum[0],fEndNum[0])
    n=data.shape[0]
    m=data.shape[1]

    #print(m,n,l)
    psd = np.zeros([n,len(fStart)])   #n对应通道数       len（fstart）对应频带数
    de = np.zeros([n,len(fStart)])
    #Hanning window
    Hlength=window*fs
    #Hwindow=hanning(Hlength)
    Hwindow= np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength+1)) for n in range(1,Hlength+1)])

    WindowPoints=fs*window
    dataNow=data[0:n]
    for j in range(0,n):  #对n个电极
        temp=dataNow[j]
        Hdata=temp*Hwindow   #加窗处理
        FFTdata=fft(Hdata,STFTN)
        magFFTdata=abs(FFTdata[0:int(STFTN/2)])  #短时傅里叶变换（FFT）转换到频域，计算出每个频点的幅度谱 magFFTdata
        for p in range(0,len(fStart)):  #每个频带
            E = 0
            #E_log = 0
            for p0 in range(fStartNum[p]-1,fEndNum[p]):  #每个频率
                E=E+magFFTdata[p0]*magFFTdata[p0]        #功率谱密度E 是频带内幅度谱的平方和的平均值
            #    E_log = E_log + log2(magFFTdata(p0)*magFFTdata(p0)+1)
            E = E/(fEndNum[p]-fStartNum[p]+1)
            psd[j][p] = E
            de[j][p] = math.log(100*E,2)  #微分熵 公式
            #de(j,i,p)=log2((1+E)^4)
    
    return de, psd

