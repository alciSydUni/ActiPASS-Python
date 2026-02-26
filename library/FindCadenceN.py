import numpy as np
import pandas as pd
from scipy.singal import butter, filtfilt, medfilt

def FindCadenceN(accDF: pd.DataFrame,actDetect: np.ndarray,SF: int) -> np.ndarray:

## ->   Thresholds
    spOther = 0.1

## ->   Filtering - using order 3 for double direction
    b25, a25 = butter(3,2.5/(SF/2),btype='low')
    xLow25 = filtfilt(b25,a25,accDF['X'])
    b15, a15 = butter(3,1.5/(SF/2),btype='high')
    xWalk = filtfilt(b15,a15,xLow25)
    b30, a30 = butter(3,3/(SF/2),btype='high')
    xRun = filtfilt(b30,a30,xWalk)

## ->   FFT preparation
    fStep, walk,run, stairs, other = (np.zeros(actDetect.shape) for _ in range(5))
    walk[actDetect==5] = 1
    run[actDetect==6] = 1
    stairs[actDetect==7] = 1
    other[actDetect==9] = 1
    combArr = walk+run+stairs+other
    N = len(accDF['X'])
    fftWin = 128
    fScale = (SF/2)*np.linspace(0,1,int(fftWin/2))

## ->   FFT calculation
    for i in range(len(actDetect)):
        if combArr[i] == 1:
            ii = np.arange(max(0,i*SF-(fftWin/2-1)),min(i*SF+fftWin/2,N-1)+1)
            if run[i] == 1:
                x = xRun[ii] - np.mean(xRun[ii])
            else:
                x = xWalk[ii] - np.mean(xWalk[ii])
            fftX = np.fft.fft(x,fftWin)
            P1 = 2*np.abs(fftX[:fftWin//2]) / fftWin
            maxP1 = np.max(P1)
            maxP1I = np.argmax(P1)
            if other[i] != 1:
                fStep[i] = fScale[maxP1I]
            else:
                if maxP1 > spOther:
                    fStep[i] = fScale[maxP1I]
    fStep = medfilt(fStep,kernel_size=3)

## ->   return value
    return fStep




