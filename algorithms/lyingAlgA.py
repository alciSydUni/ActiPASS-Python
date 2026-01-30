import ActivityDetect
import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray
import pandas as pd
import scipy.signal
import scipy.ndimage
from library import HelperFunctions
import rle


def LyingAlgA(accFilt: np.ndarray, vRefThigh: np.ndarray, actDetect: np.ndarray, SF: int) -> NDArray[np.int_]:

## -> variables
    tWinMean = 20
    tVHigh = 65
    tVLow = 64
    tVNoise = 0.05
    minLieT = 1

## ->   rotation matrix local -> global
    rot = np.array([[np.cos(vRefThigh[1]),0,np.sin(vRefThigh[1])], 
                    [0,1,0],
                    [-np.sin(vRefThigh[1]),0,np.cos(vRefThigh[1])]])
    accFiltRot = accFilt @ rot

## ->   moving mean of accFilt
    accFiltRotMMean = scipy.ndimage.uniform_filter1d(accFiltRot, size=tWinMean*SF, axis=0, mode="nearest", origin= -(tWinMean*SF//2))
    accFiltRotMMean_1S = accFiltRotMMean[0::SF,:]

## ->   find thigh rot V
    thighV = abs(np.degrees(np.asin(accFiltRotMMean_1S[:,1]/np.sqrt(accFiltRotMMean_1S[:,1]**2 + accFiltRotMMean_1S[:,2]**2))))
    thighV = np.vstack([thighV[0:1,:], thighV])

## ->   process upper/lower cutoffs for lying
    crossHighRotPt = np.diff(thighV > tVHigh) > 0
    crossLowRotPt = np.diff(thighV < tVLow) > 0
#       calculate noise & filter
    noiseIndex = np.abs(np.diff(thighV))>= tVNoise
    crossHighRotPt = crossHighRotPt & noiseIndex
    crossLowRotPt = crossLowRotPt & noiseIndex

## ->   process sit activities
    sitPts = actDetect == 2
    rleVal, rleCnt = rle.encode(sitPts)
    rleVal = np.asarray(rleVal)
    rleCnt = np.asarray(rleCnt)
    rleStartI, rleEndI = HelperFunctions.RLEIndeces(rleVal,rleCnt)
    sitSec = np.where(rleVal==1)[0]
    for sec in range(len(sitSec)):
        secRng = np.arange(rleStartI[sitSec[sec]],rleEndI[sitSec[sec]]+1)
        if len(secRng)>minLieT:
            ptsH = np.where(crossHighRotPt[secRng])[0]
            ptsL = np.where(crossLowRotPt[secRng])[0]
            if len(ptsH)>=1 and len(ptsL)>=1:
                actDetect[secRng] = 1
    
## ->   return
    return actDetect




    