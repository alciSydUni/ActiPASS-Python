import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray
from scipy.ndimage import label
from datetime import datetime, timedelta
import padas as pd

def RLEIndeces(values: np.ndarray, counts: np.ndarray) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    startIndeces = np.zeros_like(values, dtype = int)
    endIndeces = np.zeros_like(values, dtype = int)
    for i in range(len(values)):
        startIndeces[i] = sum(counts[:i])
        endIndeces[i] = startIndeces[i] + counts[i] - 1
    return startIndeces, endIndeces


def MatlabBwareaopen(arr: np.ndarray, minLength: int, removeType: bool) -> np.ndarray:
    if removeType:
        labeledArr, numFeatures = label(arr)
        outputArr = np.zeros_like(arr, dtype=bool)
        for i in range(1, numFeatures + 1):
            iRegionMask = (labeledArr == i)
            if np.sum(iRegionMask) >= minLength:
                outputArr[iRegionMask] = True
        return outputArr
    else:
        labeledArr, numFeatures = label(~arr)
        outputArr = np.zeros_like(arr, dtype=bool)
        for i in range(1, numFeatures + 1):
            iRegionMask = (labeledArr == i)
            if np.sum(iRegionMask) >= minLength:
                outputArr[iRegionMask] = True
        return ~outputArr
    

def MatlabMovMean(arr, window):
    finalArr  = np.empty_like(arr, dtype=float)
    halfWindow = window // 2 #int - floor division
    for i in range(len(arr)):
        start = max(0, i-halfWindow)
        end= min(len(arr),i+halfWindow+1)
        finalArr[i] = np.mean(arr[start:end])
    return finalArr #return shape (N,)

def MatlabDatenumToDate(dn: float) -> datetime:
    return datetime.fromordinal(int(dn)) - timedelta(days=366) + timedelta(days=dn % 1)


def ChangeAxes(Acc: pd.DataFrame, devType: str, oType: int, rangeObj: Optional[range] = None) -> pd.DataFrame:  
    if rangeObj == None:
        rangeObj = range(0, Acc.shape[0])
    accSl = Acc.iloc[rangeObj][['X','Y','Z']].copy()
    if devType.lower() in ['ActiGraph','Axivity','ActivPAL','SENS','Generic','Movisens']:
        if oType == 1:
            accSl[['X','Y','Z']] = -accSl[['X','Y','Z']]
        if oType == 2:
            accSl['X'] = -accSl['X']
        if oType == 3:
            accSl['Z'] = -accSl['Z']
        if oType == 4:
            accSl['Y'] = -accSl['Y']
    Acc.iloc[rangeObj, Acc.columns.get_indexer(['X','Y','Z'])] = accSl.values
    return Acc
            