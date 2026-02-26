import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray
import pandas as pd
import scipy.signal
import scipy.ndimage
import AktFilt

def ActivityDetect(Acc: pd.DataFrame, SF: int, tSeries: pd.Series, VRefThigh: np.ndarray, ParamsAP: dict) -> Tuple[NDArray[np.int64],NDArray[np.float64],pd.Series,pd.DataFrame]:
    tStndMv = ParamsAP["Threshold_standmove"] 
    tWlkRn = ParamsAP["Threshold_walkrun"]
    tStStnd = ParamsAP["Threshold_sitstand"]
    tStrCcl = ParamsAP["Threshold_staircycle"]

## ->   Setup time component of Acc
    lenAcc = Acc.shape[0]
    timeDtNmS = tSeries[0] + np.arange(np.floor((lenAcc-1)/SF)+1)/86400

## ->   Rotation matrix
    rotFB = np.array([
        [np.cos(VRefThigh[1]),0,np.sin(VRefThigh[1])],
        [0,1,0],
        [-np.sin(VRefThigh[1]),0,np.cos(VRefThigh[1])]
        ])

## ->   Rotate from local to global
    accRot = pd.DataFrame(Acc[['X','Y','Z']] @ rotFB, columns=['X','Y','Z'],index=Acc.index)

## ->   Perform a low pass at 5Hz
    Bl, Al = scipy.signal.butter(N=4, Wn=5, btype='low',fs=SF, analog=False)
    accRotLow = pd.DataFrame(scipy.signal.lfilter(Bl, Al, accRot[['X','Y','Z']].to_numpy(), axis=0),columns=['X','Y','Z'], index=accRot.index)

## ->   Compute a 2s moving STD and downsample to 1s intervals -> DataFrame
    accRotLowMSTD = accRotLow[['X','Y','Z']].rolling(window=2*SF, center=False).std(ddof=1)
    accRotLowMSTD_1S = accRotLowMSTD.iloc[::SF,:]
    
## ->   Adjust STD values
    if ParamsAP["ftype"] == 6:
        accRotLowMSTD_1S = 0.18 * (accRotLowMSTD_1S ** 2) + 1.03 *  accRotLowMSTD_1S

## ->   Compute a 2s moving mean and downsample to 1s intervals -> DataFrame
    accRotLowMMean = accRotLow[['X','Y','Z']].rolling(window=2*SF, center=False).mean()
    accRotLowMMean_1S = accRotLowMMean.iloc[::SF,:]

## ->   Compute length of accRotLowMMean_1S -> Data Series
    accRotLowMMean_1S_SVM = np.sqrt(accRotLowMMean_1S['X']**2 + accRotLowMMean_1S['Y']**2 + accRotLowMMean_1S['Z']**2)

## ->   Compute forward/backward tilt angles
    Inc = (180/np.pi)*np.acos(accRotLowMMean_1S['X']/accRotLowMMean_1S_SVM)
    FB = -(180/np.pi)*np.asin(accRotLowMMean_1S['Z']/accRotLowMMean_1S_SVM)
    accRotLowMSTD_1S_MAX = accRotLowMSTD_1S[['X','Y','Z']].max(axis=1)

## ->   Compute the stair climbing threshold angle 'angleStair'
    tAngle = 4
    angleStair = tAngle + np.median(FB[((0.25 < accRotLowMSTD_1S['X']) & (accRotLowMSTD_1S['X']< tWlkRn) & (FB < 25))])
#       numpy arrays:    
    row, cycle, stair, run, walk, sit, stand = [np.zeros_like(Inc) for _ in range(7)]

#       row
    row[((90 < Inc) & (tStndMv < accRotLowMSTD_1S['X']))] = 1
    row = scipy.ndimage.median_filter(row, size=2*ParamsAP["Bout_row"]-1, mode='constant', cval=0)
    row = scipy.ndimage.median_filter(row, size=2*ParamsAP["Bout_row"]-1, mode='constant', cval=0)
    etter = row

#       cycle
    maybeCycle = np.zeros_like(cycle)  
    maybeCycle[((tStrCcl-15<FB) & (Inc<90) & (tStndMv<accRotLowMSTD_1S['X']))] = 1
    cycle = CalcCycle(maybeCycle, tStrCcl, FB, Acc, SF)
    cycle = scipy.ndimage.median_filter(cycle, size=2*ParamsAP["Bout_cycle"]-1, mode='constant', cval=0)
    cycle = scipy.ndimage.median_filter(cycle, size=2*ParamsAP["Bout_cycle"]-1, mode='constant', cval=0)
    cycle = cycle * (~etter)
    etter = cycle + etter

#       stair
    stair[(angleStair<FB) & (FB<tStrCcl) & (tStndMv<accRotLowMSTD_1S['X']) & (accRotLowMSTD_1S['X']<tWlkRn) & (Inc<tStStnd)] = 1
    stair = scipy.ndimage.median_filter(stair, size=2*ParamsAP["Bout_stair"]-1, mode='constant', cval=0)
    stair = scipy.ndimage.median_filter(stair, size=2*ParamsAP["Bout_stair"]-1, mode='constant', cval=0)
    stair = stair * (~etter)
    etter = stair + etter
    
#       run
    run[(accRotLowMSTD_1S['X']>tWlkRn) & (Inc<tStStnd)] = 1
    run = scipy.ndimage.median_filter(run,2*ParamsAP["Bout_run"]-1, mode='constant', cval=0)
    run = scipy.ndimage.median_filter(run,2*ParamsAP["Bout_run"]-1, mode='constant', cval=0)
    run = run *(~etter)
    etter = run + etter

#       walk
    walk[(tStndMv<accRotLowMSTD_1S['X']) & (accRotLowMSTD_1S['X']<tWlkRn) & (FB<angleStair) & (Inc<tStStnd)] = 1
    walk = scipy.ndimage.median_filter(walk, 2*ParamsAP["Bout_walk"]-1, mode='constant', cval=0)
    walk = scipy.ndimage.median_filter(walk, 2*ParamsAP["Bout_walk"]-1, mode='constant', cval=0)
    walk = walk * (~etter)
    etter = walk + etter

#       stand
    stand[(Inc<tStStnd) & (accRotLowMSTD_1S_MAX<tStndMv)] = 1
    stand = scipy.ndimage.median_filter(stand, 2*ParamsAP["Bout_stand"]-1, mode='constant', cval=0)
    stand = scipy.ndimage.median_filter(stand, 2*ParamsAP["Bout_stand"]-1, mode='constant', cval=0)
    stand = stand * (~etter)
    etter = stand + etter

#       sit
    sit[(Inc>tStStnd)] = 1
    sit = scipy.ndimage.median_filter(sit, 2*ParamsAP["Bout_sit"]-1, mode='constant', cval=0)
    sit = scipy.ndimage.median_filter(sit, 2*ParamsAP["Bout_sit"]-1, mode='constant', cval=0)
    sit = sit * (~etter)
    etter = sit + etter

## ->   remove short bouts
    move = ~etter
    combArr = 2*sit + 3*stand + 4*move + 5*walk + 6*run + 7*stair + 8*cycle + 9*row
    combArr = AktFilt.AktFilt(combArr,'row',ParamsAP)
    combArr = AktFilt.AktFilt(combArr,'cycle',ParamsAP)
    combArr = AktFilt.AktFilt(combArr,'stair',ParamsAP)
    combArr = AktFilt.AktFilt(combArr,'run',ParamsAP)
    combArr = AktFilt.AktFilt(combArr,'walk',ParamsAP)
    combArr = AktFilt.AktFilt(combArr,'move',ParamsAP)
    combArr = AktFilt.AktFilt(combArr,'stand',ParamsAP)
    combArr = AktFilt.AktFilt(combArr,'sit',ParamsAP)

## ->   return
    return combArr, timeDtNmS, FB, accRotLowMSTD_1S


def CalcCycle(maybeCycle: np.ndarray, tStrCcl: int, FB: pd.Series, Acc: pd.DataFrame, SF: int) -> np.ndarray:
    bh, ah = scipy.signal.butter(N=3, Wn=1, btype='high', fs=SF)
    accHigh = scipy.signal.filtfilt(bh, ah, Acc['Z']) #numpy
    bl, al = scipy.signal.butter(N=3, Wn=1, btype='low', fs=SF)
    accLow = scipy.signal.filtfilt(bl, al, Acc['Z']) #numpy
    N = Acc.shape[0]
    cycle = np.zeros_like(maybeCycle, dtype=int)
    maybeCycle = scipy.ndimage.median_filter(maybeCycle, size=9, mode='constant', cval=0)

## ->   loop over maybeCycle and adjust based on additional logic
    for i in range (len(maybeCycle)):
        if maybeCycle[i]:
            start = max(0, i*SF-63)
            end = min(i*SF+64, N)
            iStartEnd = np.arange(start,end)
            hlRatio = np.mean(np.abs(accHigh[iStartEnd]))/np.mean(np.abs(accLow[iStartEnd]))
            if hlRatio < 0.5 or tStrCcl < FB.iloc[i]:
                cycle[i] = 1
    
## ->   return valye
    return cycle
