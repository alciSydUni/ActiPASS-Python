import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Dict, List, Any, Tuple
from FindAnglesAndVM import FindAnglesAndVM_DF
from ..utils.readGenericCSV import matlabDatenumArr
import math
import rle

def NotWornQC(Acc: pd.DataFrame,
    meanTEMP: np.ndarray, #shape: (len(smplsOf1S),) OR (0,)
    diaryStrct: List[Dict[str,Any]], #defined as: [{} for _ in range(len(subjectIDs))] from open_diary.py
    Fs: int,
    smplsOf1S: np.ndarray  #list: index values for Acc - Fs apart - starting from 0
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float_], List[str]]:

    ## -> variable definitions
    filtWin = 2 #time: seconds
    Fc = 2 #frequency: Hz -> cut off
    tShortWorn = 120 #time: seconds
    tEdgeWorn = 1800 #time: seconds 
    tMinNotWorn = 5400 #time: seconds
    tSeperation = 600 #time: seconds
    stillThrshld = 0.01 #STD: max size for not worn
    movThrshld = 0.1 #STD
    filtWinSz = Fs*filtWin #time: seconds
    warningStr: List[str] = [] #List[str]: warnings

    ## -> initialize vectors of size len(smplsOf1S)
    smplTimes = Acc.loc[smplsOf1S,'time']
    #initialize with one column per subjectID to give -> shape (MxN) -> (len(smplTimes),len(diaryStrct))
    #use len(diaryStrct) = 1 -> all sizes below Nx1
    notWornLogic = np.zeros((len(smplTimes), len(diaryStrct)), dtype=bool) #Nx1
    NWForce = np.zeros((len(smplTimes), len(diaryStrct)), dtype=bool) #Nx1
    nightLogic = np.zeros((len(smplTimes), len(diaryStrct)), dtype=bool) #Nx1

    ## -> get moving-STD, row-wise mean and sum
    movSTDAcc = Acc[['X','Y','Z']].rolling(window=filtWinSz, center=True,min_periods=1).std(ddof=0)
    movSTDAcc = movSTDAcc.loc[smplsOf1S,:] #len(smplsOf1S) df
    stdMean = np.mean(movSTDAcc,axis=1) #(N,)
    stdSum = np.sum(movSTDAcc,axis=1) #(N,)

    ## -> apply low-pass butter filter to acceleration
    V, accFilt, SVM, normAcc = FindAnglesAndVM_DF(Acc,Fs,Fc,smplsOf1S) #len(smplsOf1S) rows for all

    ## -> evaluate the stdMean (STD mean per row) against thresholds
    stillAccLogic = np.concatenate([np.array([False]), stdMean[:-1] < stillThrshld, np.array([False])]) #(N+1,)
    moveAccLogic = stdMean > movThrshld #(N,) = (len(smplsOf1S),)

    ## -> evaluate data from meanTEMP
    if meanTEMP.size > 0:
        lowTemp = np.concatenate([np.array([False]), meanTEMP[:-1] < np.percentile(meanTEMP[moveAccLogic],5), np.array([False])]) #(N+1,)
        stillAccLogic = np.logical_or(stillAccLogic,lowTemp) #(N+1,)

    ## -> process stillAccLogic for size of segments - 1st filtering
    #offState = True = still | onState = False = movement
    offOnChange = np.diff(stillAccLogic) #(N,)
    #offOnChange[0] IF = 1, then stillAccLogic[1] = True, because of padding, then stdMean[0] < stillThrshld 
    #+1 to align offState and onState to stdMean, which is the raw feature of stillness
    offState = np.where(offOnChange == 1)[0] + 1 #shape: (np.sum(offOnChange == 1),)
    onState = np.where(offOnChange == -1)[0] + 1 #shape: (np.sum(offOnChange == -1),)
    #len(offState) = len(onState) -> stillAccLogic is padded: (False, -arr-, False)
    #onState - offState = duration of stillness periods array
    onMinusOff = onState - offState #shape: (stillness segments,)
    #len(offState) = len(onState) = len(onMinusOff)
    #len(offStateFiltered) = len(onStateFiltered) <= offState, onState, onMinusOff
    #maintain index to segments -off- and -on- points for segments longer than tSeparation in length
    offStateFiltered = offState[onMinusOff > tSeperation] #shape: (len(onMinusOff > tSeperation),)
    onStateFiltered = onState[onMinusOff > tSeperation] #shape: (len(onMinusOff > tSeperation),)

    ## -> process for pre-stillness state OR stillness segment is long - 2nd filtering
    okStillnessLogic = np.zeros_like(offStateFiltered, dtype=bool) #shape: (len(offStateFiltered),)
    for i in range(len(okStillnessLogic)):
        okStillnessLogic[i] = np.max(stdSum[max(offStateFiltered[i]-15,0):max(offStateFiltered[i]-11,4)+1]) > 0.5 \
        or onStateFiltered[i] - offStateFiltered[i] > tMinNotWorn
    offStateFiltered = offStateFiltered[okStillnessLogic] #shape: less or equal (len(okStillnessLogic),)
    onStateFiltered = onStateFiltered[okStillnessLogic] #shape: less or equal (len(okStillnessLogic),)

    ## -> process for movement segments between stillness states - 3rd filtering
    #onStateFiltered[i] is the end of a segment and
    #offStateFiltered[i+1] is the start of the next segment
    #offStateFiltered[i+1] - onStateFiltered[i] = movement gaps boolean array
    longOn = offStateFiltered[1:] - onStateFiltered[:-1] >= tShortWorn #shape: (len(offStateFiltered)-1,) = (len(onStateFiltered)-1,)
    if longOn.size > 0: 
        onStateFiltered = onStateFiltered[np.concatenate([longOn,[True]])] #shape: <= (len(longOn)+1,)
        offStateFiltered = offStateFiltered[np.concatenate([[True],longOn])]#shape: <= (len(longOn)+1,)
    
    ## -> snap stillness segments to the beginning and end of the stillAccLogic timeline - 4th adjustment
    if offStateFiltered.size > 0 and offStateFiltered[0] < tEdgeWorn:
        offStateFiltered[0] = 0
    if onStateFiltered.size > 0 and (len(offOnChange)-onStateFiltered[-1]) < tEdgeWorn:
        onStateFiltered[-1] = len(offOnChange) - 1

    ## -> use offStateFiltered/onStateFiltered -> index smplTimes -> adjust notWornLogic
    for i in range(0,len(offStateFiltered)): #indexes smplTimes
        meanV = (180/np.pi)*np.mean(V[offStateFiltered[i]:onStateFiltered[i]+1,:],axis=0) #shape: (3,)
        if onStateFiltered[i] - offStateFiltered[i] > tMinNotWorn \
        or np.all(np.abs(meanV - np.array([90,90,0])) < 5 ) \
        or np.all(np.abs(meanV - np.array([90,-90,0])) < 5):
            notWornLogic[offStateFiltered[i]:onStateFiltered[i]] = True
            
    ## -> process diaryStruct to find forced non-wear and bed/night period
    oldEvent = "NE"
    #convert diarStrct[i]['Ticks'] to Python list of arrays
    diaryStrctTicksMD = [matlabDatenumArr(diaryStrctI['Ticks']) 
                       if 'Ticks' in diaryStrctI and diaryStrctI['Ticks'] is not None 
                       else None 
                       for diaryStrctI in diaryStrct] #List of (N,) - some values can be None
    #extract indeces for alignment of diaryStrctTicksMD[i] and smplTimes -> diaryIndex[i] and smpleTimesIndex[i]
    # diaryStrctTicksMD[i] returns a time array from diaryStrct[i]['Ticks'] 
    # these time values are checked against smplTimes
    # if matches are found
    # respective indeces in diaryStrctTicksMD[i] and smplTimes are extracted to -> diaryIndex[i], smpleTimesIndex[i]
    diaryIndex = np.empty(len(diaryStrct), dtype=object)
    smpleTimesIndex = np.empty(len(diaryStrct), dtype=object)
    currEventStrArr = np.empty((len(smplTimes),len(diaryStrct)), dtype=object) #object for flextible str length
    currEventStrArr[:] = ""
    for i in range(len(diaryStrct)): #[i] is for each subjectID in diaryStrct
        if diaryStrctTicksMD[i] is not None:
            _, diaryIndex[i], smpleTimesIndex[i] = np.intersect1d(diaryStrctTicksMD[i], \
            np.round(smplTimes.to_numpy()*86400)/86400, return_indices=True) #type: ignore
        else:
            diaryIndex[i] = np.array([], dtype=int)
            smpleTimesIndex[i] = np.array([], dtype=int)
    
        #iterate over smpleTimesIndex by subject [i] and then by diaryIndex value for diaryStrct[i]['Event']
        #eventIndices = np.unique(np.concatenate([[0], [len(smplTimes)-1],smpleTimesIndex[i]]))
        
        for k in range(len(smpleTimesIndex[i]) - 1): #(N,) where N = diaryStrct[i]['Ticks'] matches
            # the indices of diaryIndex[i] and smpleTimesIndex[i] should be aligned
            # guaranteeing that [i] and [k] exist for diaryStrct
            # propagate an 'Event' across multiple 'Ticks' if needed
            if 'Events' in diaryStrct[i] and diaryStrct[i]['Events'][diaryIndex[i][k]]:
                currEventStrArr[smpleTimesIndex[i][k],i] =  diaryStrct[i]['Events'][diaryIndex[i][k]]
                if  currEventStrArr[smpleTimesIndex[i][k],i].lower() == 'start':
                    currEventStrArr[smpleTimesIndex[i][k],i] = 'NE'
                else:
                    oldEvent = currEventStrArr[smpleTimesIndex[i][k],i]
            else:
                currEventStrArr[smpleTimesIndex[i][k],i] = oldEvent
            
            #update nightLogic, NWForce, notWornLogic
            currEvent =  currEventStrArr[smpleTimesIndex[i][k],i]
            start = smpleTimesIndex[i][k]
            end = smpleTimesIndex[i][k+1]
            if currEvent.lower() in ['night', 'bed', 'bedtime']:
                nightLogic[start:end,i] = True #last smplTimes before k+1
            if any(e.lower() == currEvent.lower() for e in ['NW','MNW','FNW','ForcedNW']):
                NWForce[start:end,i] = True
            if currEvent.lower() == 'NE'.lower() or 'leisure' in currEvent.lower():
                #determine if current time from diaryStrct[i]['Ticks'] match is at night
                fractionalDay = np.remainder(smplTimes[start:end].to_numpy(),1) #(N,)
                nightStart = 22/24
                nightEnd = 7/24
                isNightMask = (fractionalDay > nightStart) | (fractionalDay < nightEnd)
                nightLogic[start:end,i] = isNightMask #nightLogic/isNightMask defined from smplTimes
    

    if not np.any(nightLogic): #no night time flagged
        notWornLogic = notWornLogic | NWForce
    else:
        nightLogicPad = np.concatenate([[False],nightLogic,[False]]) # (N+2,1)
        nightLogicPadDiff = np.diff(nightLogicPad.astype(int))
        nightTrue = np.where(nightLogicPadDiff == 1)[0]
        nightFalse = np.where(nightLogicPadDiff == -1)[0]
        lengthOfNightPeriods = nightFalse - nightTrue #(N,) where N = night periods | value = period length
        #len(nightTrue)/len(nightFalse) = number of night segments
        #nightFalse[j] - nightTrue[j] = units of smplTimes = seconds in segment [j]

        try: #evaluate each night segment independently
            for j in range(len(nightTrue)):
                if lengthOfNightPeriods.size > 0 and \
                np.sum(notWornLogic[nightTrue[j]: nightFalse[j]+1])/lengthOfNightPeriods[j] < 0.5:
                    notWornLogic[nightTrue[j]: nightFalse[j]+1] = False
        except Exception as e:
            raise RuntimeError(f"Invalid or malformed data encountered. Aborting logic. Exception: {e}")
        notWornLogic = notWornLogic | NWForce

        wearDays = smplTimes.to_numpy()[-1] - smplTimes.to_numpy()[0] - (np.sum(notWornLogic.astype(int))/86400)
        rleValues, rleCounts  = rle.encode(notWornLogic.astype(int))
        numRegionsNW = np.sum(rleValues)
        if numRegionsNW > wearDays: #expecting < 1 NW per day
            warningStr.append("More NW segments than days. Expected 1 or less than 1 NW segment per day.")

    return notWornLogic, nightLogic, stdSum, warningStr
        
        





                    












