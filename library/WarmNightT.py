import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import List, Union, Tuple
from datetime import datetime,timedelta
from scipy.signal import medfilt
from library.QCFlipRotation import MatlabBwareaopen
import rle


def WarmNightT(
    meanTEMP: np.ndarray,
    time1SFromAccTrim: pd.Series,
    nightLogic: np.ndarray
) -> Tuple[NDArray[np.bool_], List[str]]:
    
    joinTimerActive = 2*60 #seconds
    joinTimerSleep= 45 * 60 #seconds
    tempLimit=31.7 #degrees celsius
    tShortLimit=1 #hours
    tMedFWin=60*3 #seconds
    warningStr: List[str] = [] #warning strings

    ## ->   process temperature data
    if meanTEMP.size != 0:
        #   warn if summer months    
        for d in MatlabDatenumToDatetimeArr([time1SFromAccTrim.iloc[0],time1SFromAccTrim.iloc[-1]]):            
            if d.strftime('%b') in ["Jun","Jul","Aug"]:
                warningStr.append("Summer months, possible incorrect bedtimes used for flip detection")
        #   median filtering | meanTEMP is (N,) from smplsOf1S
        meanTEMPFiltered = medfilt(meanTEMP, kernel_size=tMedFWin)
        logicWarm = meanTEMPFiltered > tempLimit
        warmNightLogic = logicWarm & nightLogic
    else:
        warmNightLogic = nightLogic
    
    ## ->   main night adjustment
    #   filter small gaps in activity and sleep
    warmNightLogic = MatlabBwareaopen(warmNightLogic,joinTimerActive,False)
    warmNightLogic = MatlabBwareaopen(warmNightLogic,joinTimerSleep,True)
    
    #   encode the values and counts and extract index values
    #   e.g. arr = [1 1 0 0 1 1] rle(arr) -> values = [1 0 1] -> counts = [2 2 2] -> sectionsWNL = [0,2]
    values, counts = rle.encode(warmNightLogic)
    values = np.array(values, dtype=int)
    counts = np.array(counts, dtype=int)
    sectionsWNL = np.where(values == 1)[0]
    startIndecesArr, endIndecesArr = RLEIndeces(values,counts)
    
    #   classify longer periods of warmNightLogic
    tooMuchSleepBool = False
    for b in range(len(sectionsWNL)):
        if counts[sectionsWNL[b]]/3600 < tShortLimit:
            warmNightLogic[startIndecesArr[sectionsWNL[b]]:endIndecesArr[sectionsWNL[b]]] = False
        if counts[sectionsWNL[b]]/3600 > 11:
            tooMuchSleepBool = True
    if tooMuchSleepBool:
        warningStr.append("Possible bedtime over-estimate for flip detection")
    
    return warmNightLogic, warningStr

def RLEIndeces(values: np.ndarray, counts: np.ndarray) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    startIndeces = np.zeros_like(values, dtype = int)
    endIndeces = np.zeros_like(values, dtype = int)
    for i in range(len(values)):
        startIndeces[i] = sum(counts[:i])
        endIndeces[i] = startIndeces[i] + counts[i] - 1
    return startIndeces, endIndeces

def MatlabDatenumToDatetime(mlDatenum: float) -> datetime:
    return datetime(1,1,1) + timedelta(days=mlDatenum-366)

def MatlabDatenumToDatetimeArr(mlDatenumArr: Union[List[float], np.ndarray]) -> NDArray[np.object_]:
    return np.array([datetime(1,1,1)+timedelta(days=d-366) for d in mlDatenumArr])
