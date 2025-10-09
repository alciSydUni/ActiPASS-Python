import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import math
from scipy.interpolate import interp1d
from scipy.ndimage import label, median_filter
from scipy.signal import medfilt
from library.NotWornQC import NotWornQC
from library.WarmNightT import WarmNightT

def QCFlipRotation(
    Acc: pd.DataFrame,
    TEMP: np.ndarray,
    diaryStrct: List[Dict[str,Any]],
    devType: str,
    Settings: tuple[dict,dict], #Settings[0] = Settings | Settings[1] = ParamsAP
    nwTrimMode: List[bool]
) -> None:

    ## ->   variable definitions
    twinFilt = 2 #time
    tshortSit = 20 #seconds
    twinMedFilt = 8 #seconds
    twinWlk = 10 #time
    nwShortLim = Settings[0]["NWSHORTLIM"]*3600 #seconds #Matlab Name: maxNWWin
    nwTrimActLim = Settings[0]["NWTRIMACTLIM"]*3600 #seconds #Matlab Name: minActiveWin
    nwTrimBuf = Settings[0]["NWTRIMBUF"]*3600 #seconds #Matlab Name: trimBuffer
    minWornT = 60 #time
    minSitT = 120 #time
    minSitTPD = 1800 #time
    minWalkT = 30 #seconds
    status = "" #str
    warnings: List[str] = [] #list
    qcData: Dict[str, Any] = {} 
    meanTEMP: np.ndarray = np.array([])

    ## ->   set default orientation from Settings[0]
    if Settings[0]["Rotated"]:
        defRotation = -1
    else:
        defRotation = 1
    
    if Settings[0]["Flipped"]:
        defFlip = -1
    else:
        defFlip = 1
    
    ## ->   set execution mode
    exMode = Settings[0]["FLIPROTATIONS"]

    try:
        ## ->   Initialize
        #       get Fs in Hz
        tmpEndInd = min(1000, Acc.shape[0])
        sampleInterval = 86400*np.mean(np.diff(Acc.iloc[:tmpEndInd]['time'].to_numpy()))
        Fs = int(round(1/sampleInterval))
        #       get last Acc index for end of last full second
        lastSmpl = math.floor(Acc.shape[0]/Fs)*Fs #index value for Acc  - last sample of the last full Fs sample
        #       smplsOf1S = list(range(0, int(lastSmpl+1), int(Fs))) #list: index values for Acc - Fs apart - starting from 0
        smplsOf1S = np.array(range(0, int(lastSmpl+1), int(Fs))) #(N,): index values for Acc - Fs apart - starting from 0
        #       TEMP
        if TEMP.size != 0:
            samplIntrvlTemp = 86400*np.mean(np.diff(TEMP[:,0])) #mean of time for 1 sample
            meanSampleWindow = np.round(60/samplIntrvlTemp) #60 samples if samplIntrvlTemp=1s
            meanTEMP = MatlabMovMean(TEMP[:,1],meanSampleWindow) #return is (N,)
            meanTEMP = np.interp(Acc.loc[smplsOf1S, 'time'].to_numpy(),TEMP[:,0], meanTEMP) #shape: (len(smplsOf1S),) | note: assume Acc 'time' stays within TEMP[:,0]
        else:
            warnings.append("No temperature data. Accuracy reduced in NW and auto flip/rotation.")

        ## ->   Call NotWornQC to get not-worn details
        notWornLogic, nightLogic, stdSum, warningStr = NotWornQC(Acc, meanTEMP, diaryStrct,Fs,smplsOf1S)
        warnings.extend(warningStr)

        ## ->   Trim data based on NW
        if np.any(nwTrimMode):
            notWornLogicFiltered = MatlabBwareaopen(notWornLogic, nwShortLim, removeType=True) #remove short True segments less than maxNWWin
            notWornLogicFiltered = MatlabBwareaopen(notWornLogicFiltered, nwTrimActLim, removeType=False) #remove short False segments less than minActiveWin
            if not all(notWornLogicFiltered): #some False exists = activity
                if nwTrimMode[0]:
                    activStart = np.where(~notWornLogicFiltered)[0][0] #activStart a scalar for index of smplsOf1S
                    activStartDayIndex = np.where(Acc.loc[smplsOf1S,'time'] >= np.floor(Acc.loc[smplsOf1S[activStart],'time']))[0][0] #activeStartDayIndex a scalar for index of smplsOf1S
                    activStart = max(0,activStartDayIndex,activStart-nwTrimBuf)
                else:
                    activStart = 0
                if nwTrimMode[1]:
                    activEnd = np.where(~notWornLogicFiltered)[0][-1]
                    activEndDayIndex = np.where(Acc.loc[smplsOf1S,'time'] < np.ceil(Acc.loc[smplsOf1S[activEnd],'time']))[0][-1]
                    activEnd = min(len(notWornLogicFiltered)-1,activEndDayIndex,activEnd+nwTrimBuf)
                else:
                    activEnd = len(notWornLogicFiltered) - 1 #same as: len(smplsOf1S) - 1
            else:
                activStart = 0
                activEnd = min(86400 - 1, len(smplsOf1S)-1)
            if activStart > 0 or activEnd < len(notWornLogicFiltered) - 1:
                qcData["cropStart"] = round(activStart/3600,2)
                qcData["cropEnd"] = round(((len(notWornLogicFiltered)-1) - activEnd)/3600,3)
                if activStart > 0 and activEnd == len(notWornLogicFiltered) -1:
                    warnings.append(f"{qcData['cropStart']} hrs of non-wear at the beginning have been removed")
                elif activStart == 0 and activEnd < len(notWornLogicFiltered) - 1:
                    warnings.append(f"{qcData['cropEnd']} hrs of non-wear at the end have been removed")
                elif activStart > 0 and activEnd < len(notWornLogicFiltered) - 1:
                    warnings.append(f"{qcData['cropStart']} and {qcData['cropEnd']} hrs of non-wear from the beginning and end have been removed")
                #   trim notWornLogic with [activStar:activEnd] indexing smplsOf1S
                notWornLogic = notWornLogic[activStart:activEnd+1,:]
                #   trim nightLogic with [activStart:activEnd]
                nightLogic = nightLogic[activStart:activEnd+1,:]
                #   trim stdSum with [activStart:activEnd]
                stdSum = stdSum[activStart:activEnd+1]
                #   trim Acc with [activStart:activEnd]
                Acc = Acc.iloc[smplsOf1S[activStart]:smplsOf1S[activEnd]+Fs]
                #   save activStart, activEnd indices
                qcData["indexActivStart"] = smplsOf1S[activStart]
                qcData["indexActivEnd"] = smplsOf1S[activEnd]
                #   trim smplsOf1S arr
                smplsOf1S = smplsOf1S[activStart:activEnd+1] - smplsOf1S[activStart]
                #   trim meanTEMP arr if it exists
                if 'meanTEMP' in locals():
                    meanTEMP = meanTEMP[activStart:activEnd+1]
        
        ## ->   define time vector
        #       save trimmed smplsOf1S
        #       save time trimmed time length            
        time1SFromAccTrim = Acc.loc[smplsOf1S,'time']
        qcData['smplsOf1S'] = smplsOf1S #save smplsOf1S arr to dict
        qcData['totalTime'] = len(smplsOf1S) #number of seconds

        #       IF no auto trimming SET trimming data to zero
        if 'cropStart' not in qcData:
            qcData['cropStart'] = 0
        if 'cropEnd' not in qcData:
            qcData['cropEnd'] = 0
        
        ## ->   check execution mode and execute flips/rotation logic
        if exMode.lower() in ['warn','force']:
            filtWinSz = Fs*twinFilt #samples for 2 seconds
            movMeanAcc = Acc.loc[:,['x','y','z']].rolling(window=filtWinSz,min_periods=1,center=True).mean()
            movMeanAcc = movMeanAcc[smplsOf1S,:]
        #       for twinMedFilt even values: Matlab is left skewed and Python right skewed: so for value 8: 4,x,3 vs 3,x,4
            medfiltXY = median_filter(np.sqrt(movMeanAcc.loc[:,'x']**2 + movMeanAcc.loc[:,'y']**2),size=twinMedFilt, mode="reflect")
        #       meanTEMP & nightLogic sampled by smplsOf1S for [activStart:activEnd] range
            nightLogic, warningStr = WarmNightT(meanTEMP,time1SFromAccTrim,nightLogic)
            warnings.extend(warningStr)








                



    
    except Exception as e:
        print()


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
    halfWindow = window // 2 #int
    for i in range(len(arr)):
        start = max(0, i-halfWindow)
        end= min(len(arr),i+halfWindow+1)
        finalArr[i] = np.mean(arr[start:end])
    return finalArr #return shape (N,)