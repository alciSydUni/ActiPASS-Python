import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import math
from scipy.interpolate import interp1d

def QCFlipRotation(
    Acc: pd.DataFrame,
    TEMP: np.ndarray,
    diaryStrct: List[Dict[str,Any]],
    devType: str,
    Settings: tuple[dict,dict], #Settings[0] = Settings | Settings[1] = ParamsAP
    nwTrimMode: List[bool]
) -> None:

    ## - > variable definitions
    twinFilt = 2 #time
    tshortSit = 20 #seconds
    twinMedfilt = 8 #seconds
    twinWlk = 10 #time
    axNWWin = Settings[0]["NWSHORTLIM"]*3600 #seconds
    inActiveWin = Settings[0]["NWTRIMACTLIM"]*3600 #seconds
    rimBuffer = Settings[0]["NWTRIMBUF"]*3600 #time
    inWorn = 60 #time
    inSitT = 120 #time
    inSitTPD = 1800 #time
    inWalkT = 30 #seconds
    status = "" #str
    warnings: List[str] = [] #list
    QCData: Dict[str, Any] = {} 
    meanTEMP: np.ndarray = np.array([])

    ## -> set default orientation from Settings[0]
    if Settings[0]["Rotated"]:
        defRotation = -1
    else:
        defRotation = 1
    
    if Settings[0]["Flipped"]:
        defFlip = -1
    else:
        defFlip = 1
    
    ## - > set execution mode
    exMode = Settings[0]["FLIPROTATIONS"]

    try:
        ## -> Initialize
        #get Fs in Hz
        tmpEndInd = min(1000, Acc.shape[0])
        sampleInterval = 86400*np.mean(np.diff(Acc.iloc[:tmpEndInd]['time'].to_numpy()))
        Fs = int(np.round(1/sampleInterval))
        #get last Acc index for end of last full second
        lastSmpl = math.floor(Acc.shape[0]/Fs)*Fs #index value for Acc  - last sample of the last full Fs sample
        smplsOf1S = list(range(0, int(lastSmpl), int(Fs))) #list: index values for Acc - Fs apart - starting from 0
        #TEMP
        if TEMP.size != 0:
            samplIntrvlTemp = 86400*np.mean(np.diff(TEMP[:,0])) #mean of time for 1 sample
            meanSampleWindow = np.round(60/samplIntrvlTemp) #60 samples if samplIntrvlTemp=1s
            meanTEMP = matlabMovMean(TEMP[:,1],meanSampleWindow) #return is (N,)
            meanTEMP = np.interp(Acc.loc[smplsOf1S, 'time'].to_numpy(),TEMP[:,0], meanTEMP) #shape: (len(smplsOf1S),) | note: assume Acc 'time' stays within TEMP[:,0]
        


    except Exception as e:
        print()

def matlabMovMean(arr, window):
    finalArr  = np.empty_like(arr, dtype=float)
    halfWindow = window // 2
    for i in range(len(arr)):
        start = max(0, i-halfWindow)
        end= min(len(arr),i+halfWindow+1)
        finalArr[i] = np.mean(arr[start:end])
    return finalArr #return shape (N,)