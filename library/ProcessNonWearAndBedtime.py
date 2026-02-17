import numpy as np
import pandas as pd
from library import HelperFunctions
import NotWorn

def ProcessNonWearAndBedtime(firstDay: bool,vThigh: np.ndarray,accDF: pd.DataFrame,SF: int,accTimeDtNm: pd.Series,actDetect: np.ndarray,diaryStrctFirst: dict,Settings: dict) -> np.ndarray:

## ->   define persistent variable
    if not hasattr(ProcessNonWearAndBedtime,"oldEvent"):
        ProcessNonWearAndBedtime.oldEvent = ""

## ->   setup nightLogic and nwForce masks
    nightLogic = np.zeros(len(accTimeDtNm),dtype=bool)
    nwForce = np.zeros(len(accTimeDtNm),dtype=bool)

## ->   match diary Ticks -to- accTimeDtNm
    _, diaryTicksI, accTimeDtNmI = np.intersect1d(HelperFunctions.matlabDatenumArr(diaryStrctFirst["Ticks"]), np.round(accTimeDtNm.values*86400)/86400, return_indices=True)
    eventI = np.unique(np.concatenate([accTimeDtNmI,[0],[len(accTimeDtNm)-1]])) #unique() is only guarding against the first/last values - accTimeDtNmI is unique already

## ->   set for first day
    if firstDay:
        ProcessNonWearAndBedtime.oldEvent = "NE"

## ->   remove short gaps
    if len(actDetect) > 0 and Settings["NWCORRECTION"].lower() == 'lying':
        lenLieFilt = Settings["BDMINLIET"]*60
        lenActFilt = Settings["BDMAXAKTT"]*60
        bedLogic = HelperFunctions.MatlabBwareaopen(actDetect==1, lenActFilt, False)
        bedLogic = HelperFunctions.MatlabBwareaopen(bedLogic,lenLieFilt,True)

## ->   interate over segments in eventI
    for itrEvent in range(len(eventI)-1):
        accTDNI = np.where(accTimeDtNmI==eventI[itrEvent])[0] #redundant line - 'eventI' is built from 'accTimeDtNmI'
        if accTDNI.size > 0:
            accTDNI = accTDNI[-1] #matlab is incorrect - there is alwatays only 1 match, because intersect() only returns deduplicated first matches for either side
            currEvent = diaryStrctFirst["Events"][diaryTicksI[accTDNI]] #matlab is incorrect - it is not possible to pull the last event for a same second multi-event, because intersect() does not index beyond first instance
        else:
            currEvent = ProcessNonWearAndBedtime.oldEvent
        if  currEvent.lower() == 'start':
            currEvent = "NE"
        else:
            ProcessNonWearAndBedtime.oldEvent = currEvent

## ->   process segment
        eventISeg = np.arange(eventI[itrEvent],eventI[itrEvent+1]+1)
        if currEvent.lower() in {'night','bed','bedtime'}:
            nightLogic[eventISeg] = True
            #if Settings["LIEALG"] == 'diary':
                #actDetect cannot be calculated - enventISeg index and actDetect resolution mismatch
        elif currEvent.lower() in {'nw','mnw','fnw','forcednw'}:
            nwForce[eventISeg] = True
        #elif actDetect.size>0 and Settings['NWCORRECTION'] in {'lying','extra'} and Settings['LIEALG'] in {'algA','algB'}:
            #nightLogic cannot be calculated, eventISeg index and bedLogic resolution mismatch      
        elif currEvent.lower() == 'ne' or 'leisure' in currEvent.lower():
            T = np.remainder(accTimeDtNm[eventISeg],1)
            nightStart = 22/24
            nightEnd = 8/24
            nightT = (T>nightStart) | (T<nightEnd)
            nightLogic[eventISeg[nightT]] = True

## ->   estimation of not worn periods
    nwArr = NotWorn.NotWorn(vThigh,accDF,SF)

## ->   consider nightLogic cases
    if np.all(~nightLogic):
        nwArr = nwArr | nwForce
    else:
        diffNL = np.diff(np.concatenate(([False],nightLogic,[False])))
        nightS = np.where(diffNL==1)[0]
        nightE = np.where(diffNL==-1)[0]-1
        lenNight = np.zeros(len(nightS),dtype=int)
        for i in range(len(nightS)):
            lenNight[i] = nightE[i] - nightS[i] + 1
            if Settings['NWCORRECTION'].lower() == 'extra':
                nwArr[nightS[i]:nightE[i]+1] = False
            else:
                if np.sum(nwArr[nightS[i]:nightE[i]+1])/lenNight[i]<0.5:
                    nwArr[nightS[i]:nightE[i]+1] = False
        nwArr = nwArr | nwForce

## ->   update actDetect and return
    actDetect[nwArr] = 0
    return actDetect
