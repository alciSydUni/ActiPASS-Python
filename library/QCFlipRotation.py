import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import math
from scipy.interpolate import interp1d
from scipy.ndimage import label, median_filter
from scipy.signal import medfilt, butter, lfilter
from scipy import stats
from library.NotWornQC import NotWornQC
from library.WarmNightT import WarmNightT
import rle
from HelperFunctions import RLEIndeces, MatlabBwareaopen, MatlabMovMean, MatlabDatenumToDate, ChangeAxes
import traceback

def QCFlipRotation(
    Acc: pd.DataFrame,
    TEMP: np.ndarray,
    diaryStrct: List[Dict[str,Any]],
    devType: str,
    Settings: tuple[dict,dict], #Settings[0] = Settings | Settings[1] = ParamsAP
    nwTrimMode: List[bool]
) -> Tuple[Dict[str,Any],pd.DataFrame,np.ndarray,str,List[str]]:

## ->   variable definitions
    twinFilt = 2 #seconds - length of moving mean window on Acc
    tshortSit = 20 #seconds - min sit period
    twinMedFilt = 8 #seconds - length of moving median window on movMeanAcc
    twinWlk = 10 #seconds - min walking period
    nwShortLim = Settings[0]["NWSHORTLIM"]*3600 #seconds #Matlab Name: maxNWWin - remove for =< nwShortLim
    nwTrimActLim = Settings[0]["NWTRIMACTLIM"]*3600 #seconds #Matlab Name: minActiveWin - remove for =< nwTrimActLim
    nwTrimBuf = Settings[0]["NWTRIMBUF"]*3600 #seconds #Matlab Name: trimBuffer - buffer before and after activ period for crop non-wear
    minWornT = 60 #seconds - min worn sum of seconds for a seg to be classified as isWornSegment
    minSitT = 120 #seconds - min sit period in 4-hr seg
    minSitTPD = 1800 #seconds - min sit period in 4-hr seg
    minWalkT = 30 #seconds - min walking per day for rotation detection
    status = "" #str - status flag
    warnings: List[str] = [] #list - warnings
    qcData: Dict[str, Any] = {} #str - hold multiple results
    meanTEMP: np.ndarray = np.array([]) #temp

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
        sampleIntervalAcc = 86400*np.mean(np.diff(Acc.iloc[:tmpEndInd]['time'].to_numpy()))
        Fs = int(round(1/sampleIntervalAcc))
    #       get last Acc index for end of last full second
        lastSmpl = math.floor(Acc.shape[0]/Fs)*Fs
    #       get index values for Acc - Fs apart - starting from 0 - ending on last full sample
        smplsOf1S = np.array(range(0, int(lastSmpl+1), int(Fs))) #(N,)
    #       process TEMP arr
        if TEMP.size != 0:
            sampleIntervalTemp = 86400*np.mean(np.diff(TEMP[:,0]))
    #           get number of samples for a 60s time window - e.g. 60 samples if 1s/sample
            meanSampleWindow = np.round(60/sampleIntervalTemp)
    #           get moving mean over meanSampleWindow        
            meanTEMP = MatlabMovMean(TEMP[:,1],meanSampleWindow)
    #`          calculate interpolated values to Acc 'time' based on smplsOf1S index
    #           shape: (len(smplsOf1S),) | note: assume Acc 'time' stays within TEMP[:,0]
            meanTEMP = np.interp(Acc.loc[smplsOf1S, 'time'].to_numpy(),TEMP[:,0], meanTEMP)
        else:
            warnings.append("No temperature data. Accuracy reduced in NW and auto flip/rotation.")

    ## ->   Call NotWornQC to get not-worn details
    #       HELP
    #       ====
    #       notWornLogic shape: (len(smplsOf1S), len(diaryStrct))
    #       nightLogic shape: (len(smplsOf1S), len(diaryStrct))
    #       stdSum shape: (len(smplsOf1S),)
    #       ====
        notWornLogic, nightLogic, stdSum, warningStr = NotWornQC(Acc, meanTEMP, diaryStrct, Fs, smplsOf1S)
        warnings.extend(warningStr)

    ## ->   Trim data based on non-wear | <F#> = Filter
        if np.any(nwTrimMode):
    #           <F1-1> TRIM MIDDLE SHORTS: remove short 'True' segments less than nwShortLim - Matlab: maxNWWin
            notWornLogicFiltered = MatlabBwareaopen(notWornLogic, nwShortLim, removeType=True)
    #           <F1-2> TRIM MIDDLE SHORTS: remove short 'False' segments less than nwTrimActLim - Matlab: minActiveWin
            notWornLogicFiltered = MatlabBwareaopen(notWornLogicFiltered, nwTrimActLim, removeType=False)
    
    #           check for some 'False' = activity
            if not all(notWornLogicFiltered):
    #               <F2-1> TRIM START create -smplsOf1S index - optional
                if nwTrimMode[0]:
    #                   first worn index (smplsOf1S)
                    activStart = np.where(~notWornLogicFiltered)[0][0]
    #                   first day index (smplsOf1S)
                    activStartDayIndex = np.where(Acc.loc[smplsOf1S,'time'] >= np.floor(Acc.loc[smplsOf1S[activStart],'time']))[0][0] #activeStartDayIndex a scalar for index of smplsOf1S
    #                   select activity start index (smplsOf1S) | add buffer IF possible
                    activStart = max(0,activStartDayIndex,activStart-nwTrimBuf)
                else:
    #                   no trimming
                    activStart = 0
    #               <F2-2> TRIM END create - smplsOf1S index - optional
                if nwTrimMode[1]:
                    activEnd = np.where(~notWornLogicFiltered)[0][-1]
                    activEndDayIndex = np.where(Acc.loc[smplsOf1S,'time'] < np.ceil(Acc.loc[smplsOf1S[activEnd],'time']))[0][-1]
                    activEnd = min(len(notWornLogicFiltered)-1,activEndDayIndex,activEnd+nwTrimBuf)
                else:
                    activEnd = len(notWornLogicFiltered) - 1
    #           all 'True' = not worn = no activity | keep first day of data
            else:
                activStart = 0
                activEnd = min(86400 - 1, len(smplsOf1S)-1)
    
    #           <F2-3> TRIM APPLY start, end or both                  
            if activStart > 0 or activEnd < len(notWornLogicFiltered) - 1:
    #               save crop amounts in hours
                qcData['cropStart'] = round(activStart/3600,2)
                qcData['cropEnd'] = round(((len(notWornLogicFiltered)-1) - activEnd)/3600,3)
    #               update warnings
                if activStart > 0 and activEnd == len(notWornLogicFiltered) - 1:
                    warnings.append(f"{qcData['cropStart']} hrs of non-wear at the beginning have been removed")
                elif activStart == 0 and activEnd < len(notWornLogicFiltered) - 1:
                    warnings.append(f"{qcData['cropEnd']} hrs of non-wear at the end have been removed")
                elif activStart > 0 and activEnd < len(notWornLogicFiltered) - 1:
                    warnings.append(f"{qcData['cropStart']} and {qcData['cropEnd']} hrs of non-wear from the beginning and end have been removed")
        
    #              trim notWornLogic
                notWornLogic = notWornLogic[activStart:activEnd+1,:]
    #              trim nightLogic
                nightLogic = nightLogic[activStart:activEnd+1,:]
    #              trim stdSum
                stdSum = stdSum[activStart:activEnd+1]
    #              trim Acc
                Acc = Acc.iloc[smplsOf1S[activStart]:smplsOf1S[activEnd]+Fs]
    #              save activStart, activEnd, activStartAcc, activEndAcc
                qcData['activStart'] = activStart
                qcData['activEnd'] = activEnd
                qcData["activStartAcc"] = smplsOf1S[activStart]
                qcData["activEndAcc"] = smplsOf1S[activEnd] + Fs - 1
    #              trim smplsOf1S 
                smplsOf1S = smplsOf1S[activStart:activEnd+1] - smplsOf1S[activStart]
    #              trim meanTEMP
                if meanTEMP.size !=0:
                    meanTEMP = meanTEMP[activStart:activEnd+1]
        
    #       define trimmed 'Acc' 'time'       
        time1SFromAccTrim = Acc.iloc[smplsOf1S]['time']
    #       save smplsOf1S
        qcData['smplsOf1S'] = smplsOf1S 
        qcData['totalTime'] = len(smplsOf1S)

    #       set 'cropStart' and 'cropEnd' to 0 if unused
        if 'cropStart' not in qcData:
            qcData['cropStart'] = 0
        if 'cropEnd' not in qcData:
            qcData['cropEnd'] = 0
        
    ## ->   check execution mode and execute flips/rotation logic
        if exMode.lower() in ['warn','force']:
    #           set moving window size | samples
            filtWinSz = Fs*twinFilt
    #           get moving mean for Acc | take smplsOf1S
            movMeanAcc = Acc.loc[:,['x','y','z']].rolling(window=filtWinSz,min_periods=1,center=True).mean()
            movMeanAcc = movMeanAcc.iloc[smplsOf1S,:]
    #           get moving median X-Y from movMeanAcc | Matlab/Python for even values Left/Right skewed
            medfiltXY = median_filter(np.sqrt(movMeanAcc.loc[:,'x']**2 + movMeanAcc.loc[:,'y']**2),size=twinMedFilt, mode="reflect")
    #           update nightLogic with warm night periods | exclude warm night periods from flip detection
            nightLogic, warningStr = WarmNightT(meanTEMP,time1SFromAccTrim,nightLogic)
            warnings.extend(warningStr)
    
    ## ->       MAIN flips/rotations detection code
    #           set arrays for mean values for x(walking), y(walking) and z(sitting)
            walkMarkerX = np.full((len(smplsOf1S),1),np.nan)
            walkMarkerY = np.full((len(smplsOf1S),1),np.nan)
            sitMarkerZ = np.full((len(smplsOf1S),1),np.nan)
    #          separate worn sections to process for flip/rotation
            rleWornValues, rleWornCounts = rle.encode(~notWornLogic)
            rleWornValues = np.array(rleWornValues)
            rleWornCounts = np.array(rleWornCounts)
    #           index 'True' values in rleWornValues
            wornPeriods = np.where(rleWornValues)[0]
    #           get 'True' periods count 
            numPeriods = len(wornPeriods)
            rleWornStartIndeces, rleWornEndIndeces = RLEIndeces(rleWornValues,rleWornCounts)
    #           store worn periods start/end times
            wornTimes = np.zeros(numPeriods * 2, dtype=float)
    #           time of 90° rot
            timeRot90 = 0
        
    ## ->       iterate worn periods
            for p in range(numPeriods):
    #               index start/end worn periods in 'notWornLogic' | smplsOf1S index
                selectRange = range(rleWornStartIndeces[wornPeriods[p]],rleWornEndIndeces[wornPeriods[p]]+1)
    #               index start/end worn periods in 'Acc'
                selectIndecesAcc = smplsOf1S[selectRange]
    #               start/end times for worn periods using 'Acc' 'time'       
                wornTimes[2*p] = Acc.iloc[selectIndecesAcc[0]]['time']
                wornTimes[2*p+1]= Acc.iloc[selectIndecesAcc[-1]]['time']
        
    ## ->           find rotations using walking detection
    #               get square-vector-magnitude
                svmWalk = np.sqrt((Acc.iloc[selectIndecesAcc][['X','Y','Z']]**2).sum(axis=1))
    #               filter sqaure-vector-magnitude with 1-3Hz bandpass
                B, A = butter(N=6,Wn=[1.0/(Fs/2), 3.0/(Fs/2)],btype='bandpass') #type: ignore
                svmWalkFiltered = lfilter(B, A, svmWalk, axis=0)
    #               create buckets from 'svmWalk' data
                buktSize = int(Fs*twinWlk)
                buktsNum = int(np.floor(len(svmWalk)/buktSize))
    #               index for last sample of last bucket
                lastSmpl = buktSize * buktsNum - 1
    #               arrange 'svmWalkFiltered' in buckets 
                svmWalkFilteredToBukts = svmWalkFiltered[:lastSmpl+1].reshape(buktSize,buktsNum)
    
    ## ->           FFT - fast-fourier-transform
    #               set frequency scale   
                fScale = Fs*np.arange((buktSize//2)+1)/buktSize      
    #               compute FFT | (buktSize,buktsNum)
                fftBukts = np.fft.fft(svmWalkFilteredToBukts, axis=0)
    #               magnitude normalized | (buktSize,buktsNum)
                P1 = np.abs(fftBukts/buktSize)
    #               take first half | mirror values
                P1 = P1[:buktSize//2+1,:]
    #               double the power
                P1[1:-1,:] = 2*P1[1:-1,:]
    #               max power per bucket | (buktsNum,)
                fPwr = np.max(P1,axis=0) 
    #               index where max power occurs along buktSize | (buktsNum,)
                fIndx = np.argmax(P1, axis=0) 
    #               covert index to freq using 'fScale' | (buktsNum,)
                fIndxFreq = fScale[fIndx]
        
    #               find buckets matching walking profile | (buktsNum,)
                wlkBukts = (fPwr > 0.3) & (fIndxFreq > 1.3) &  (fIndxFreq < 2.3)
    #               repeat each value 'buktSize' times | (buktsNum * buktSize,) | (svmWalkFilteredToBukts,)
                wlkLogic = np.repeat(wlkBukts, buktSize)
    
    ## ->           find flips using X-axis values and expected range
    #               RLE of 'wlkLogic' | 'wlkLogicCounts' = 'buktSize'
                wlkLogicValues, wlkLogicCounts = map(lambda x: np.asarray(x, dtype=np.int_), rle.encode(wlkLogic))
                wlkLogicStartIndx, wlkLogicEndIndx = RLEIndeces(wlkLogicValues, wlkLogicCounts)
    #               index into 'wlkLogicValues'
                wlkLogicSec = np.where(wlkLogicValues == 1)[0]
    #               get 'Acc' indeces | (len(wlkLogicSec),)
                wlkSecStartsWP = selectIndecesAcc[wlkLogicStartIndx[wlkLogicSec]]
                wlkSecEndsWP = selectIndecesAcc[wlkLogicEndIndx[wlkLogicSec]]
    #               iterate over 'wlkLogicSec'
                for wLSec in range(len(wlkLogicSec)):
    #                   get 'Acc' start/end indeces
                    wLSecAccI = range(wlkSecStartsWP[wLSec], wlkSecEndsWP[wLSec]+1)
    #                   get 'smplsOf1S' index values
                    wLSecSOf1SStart = np.where(smplsOf1S >= wLSecAccI.start)[0][0]
                    wLSecSOf1SEnd = np.where(smplsOf1S <= wLSecAccI.stop-1)[0][-1]
    #                   calculate mean of X and Y for 'wLSec'
                    walkMarkerX[wLSecSOf1SStart:wLSecSOf1SEnd,1] = np.mean(Acc.iloc[np.array(wLSecAccI)]['X'])
                    walkMarkerY[wLSecSOf1SStart:wLSecSOf1SEnd,1] = np.mean(Acc.iloc[np.array(wLSecAccI)]['Y'])
    #               calculate the mean across all values | use to determine rot/flip state        
#   #               ATTENTION
    #               =========
    #               nanMeanX and nanMeanY are calculated from entire range of 'walkMarkerX' and 'walkMarkerY'.
    #               Reassignment of Acc 'X' and 'Y' below is on a per worn period using 'selectIndecesAcc'.
    #               Reassignment of walkMarkerX and walkMarkerY below is not per worn period.
    #               =========
                nanMeanX = np.nanmean(walkMarkerX)
                nanMeanY = np.nanmean(walkMarkerY)
    
    ## ->           test for 90° rot        
    #               abs(nanMeanX-1) >= 0.4 --> nanMeanX <= 0.6 OR >= 1.4 
    #               abs(nanMeanX-1) <= 1.6 --> nanMeanX <= 2.6 OR >= -0.6
                if abs(nanMeanX-1) >= 0.4 and abs(nanMeanX-1) <= 1.6:
#   #                   DIFFERENCE
    #                   ==========                   
    #                   This problem is related to how the X,Y,Z axis are rotated in 3D space and which side the 
    #                   accelerometer is mounted, in order for the calculations to be correct.
    #                   ==========
    #                   check if 'nanMeanY' near 1G                    
                    if abs(nanMeanY-1) < 0.35:           
    #                       LEFT thigh mount | 90° CCW rot | adjust current worn period
                        Acc.loc[selectIndecesAcc, ['X','Y']] = np.column_stack([Acc.loc[selectIndecesAcc, 'Y'], -Acc.loc[selectIndecesAcc, 'X']])
                        walkMarkerX = walkMarkerY
                        timeRot90 = timeRot90 + len(selectRange)
    #                   check if 'nanMeanY' near -1G
                    elif nanMeanY < -0.65 and nanMeanY > -1.35:
    #                       RIGHT thigh mount | 90° CCW rot | adjust current worn period                  
                        Acc.loc[selectIndecesAcc, ['X','Y']] = np.column_stack([-Acc.loc[selectIndecesAcc,'Y'], Acc.loc[selectIndecesAcc,'X']])
                        walkMarkerX=-walkMarkerY
                        timeRot90 = timeRot90 + len(selectRange)
#   #                   ===END
    #               ===END      
          
    ## ->           find flips from sit periods
    #               'stdSum' - low movement | '~nightLogic' - day time | 'medfiltXY' - some G to Z-axis | subset of 'smplsOf1S'
                sitSelect = (stdSum[selectRange] < 0.1) and ~nightLogic[selectRange] and medfiltXY[selectRange] <= 0.7
    #               remove short sit segments
                sitSelect = MatlabBwareaopen(sitSelect,tshortSit,True)
    #               RLE of 'sitSelect'
                sitSelValues, sitSelCounts = rle.encode(sitSelect)
                sitSelValues = np.array(sitSelValues, dtype=bool)
                sitSelCounts = np.array(sitSelCounts, dtype=int)
    #               index into subset of 'smplsOf1S'
                sitSelStartI , sitSelEndI = RLEIndeces(sitSelValues, sitSelCounts)
    #               index into 'sitSelStartI' | 'sitSelEndI'
                sitSections = np.where(sitSelValues)[0]
    #               get 'smplsOf1S' indeces
                sSecSO1SStarts = np.array(selectRange)[sitSelStartI[sitSections]]
                sSecSO1SEnds = np.array(selectRange)[sitSelEndI[sitSections]]
    
    ## ->           iterate 'sitSections' | get mean for Z-axis | apply to all sec elements
                for sitSec in range(len(sitSections)):
                    sSecAccStartI = smplsOf1S[sSecSO1SStarts[sitSec]]
                    sSecAccEndI = smplsOf1S[sSecSO1SEnds[sitSec]]
                    sitMarkerZ[sSecSO1SStarts[sitSec]:sSecSO1SEnds[sitSec] + 1, 1] = Acc.iloc[sSecAccStartI:sSecAccEndI + 1]['Z'].mean(axis=0)
            
    ## ->       get 4h segments for flip/rot correction
    #           time1SFromAccTrim = Acc.loc[smplsOf1S,'time']
            startTime = time1SFromAccTrim.iloc[0]
            endTime = time1SFromAccTrim.iloc[-1]
#   #           ATTENTION
    #           =========
    #           The following conditions are assuming that if the starTime
    #           starts in the middle of a 4hr period, we would snap to the upper bound of that period
    #           but if it starts on exactly a 4 hr period, then it snaps to the end of the new 4hr period
    #           or the endTime, whichever is smaller. Shouldn't this check be performed for the IF 
    #           side as well in terms of a minimum between 'np.ceil(startTime*6)/6' and endTime ?
    #           =========
    #           get end pt for first 4 hr period
            if (startTime*6) != np.ceil(startTime*6):
                first4hEndPt = np.ceil(startTime*6)/6 
            else:
    #               why use endTime as an option ? | role of 'first4hEndPt' may reverse
                first4hEndPt = min(startTime + 1/6,endTime)
            if (endTime*6) != np.floor(endTime*6):
                last4hStartPt = np.floor(endTime*6)/6
            else:
    #               why use endTime as an option ? | role of 'last4hStartPt' may reverse
                last4hStartPt = max(endTime - 1/6, startTime)
#   #           ===END
    #           create 4hr-interv arr | snap to second 
            segMarkers = np.round(np.arange(first4hEndPt, last4hStartPt + 1e-10, 1/6)*86400)/86400
    #           combine 'Acc' 'time' values
            qcData['mTimes'] = np.concatenate(([startTime], segMarkers, [endTime]))          
            qcData['mTimes'] = np.unique(np.concatenate((qcData['mTimes'],wornTimes)))
    #           calculate 'Acc' 'time' segments
            numSegments = len(qcData['mTimes']) - 1
        
    ## ->       initialize qcData fields to hold various details
            qcData['wornTime'] = 0 #seconds
    #           save period for rotated state
            qcData['rotTime'] = timeRot90 #seconds
    #           save period for flipped state
            qcData['flipTime'] = 0 #seconds
    #           save seconds up to an 'mTimes' point
            qcData['elapsedSecs'] = np.zeros((len(qcData['mTimes']),1)) #seconds
    #           save probable rot (-1) | no-rot (+1)
            qcData['xFlipValue'] = np.zeros((numSegments,1))
    #           save probable flip (-1) | no-flip (+1)
            qcData['zFlipValue'] = np.zeros((numSegments,1))
    #           save calc rot (-1) | no-rot (+1)
            qcData['xFlipCalc'] = np.zeros((numSegments,1))
    #           save calc flip (-1) | no-flip (+1)
            qcData['zFlipCalc'] = np.zeros((numSegments,1))
    #           save period for determining probable rot
            xFlipCalcLength = np.zeros((numSegments,1)) #seconds
    #           save period for determining probable flip
            zFlipCalcLength = np.zeros((numSegments,1)) #seconds
    #           save segment worn | not-worn state       
            qcData['isWornSeg'] = np.full((numSegments,1), False)
            qcData['walkMarkerX'] = walkMarkerX
            qcData['sitMarkerZ'] = sitMarkerZ

    ## ->       calculate if 4h seg are flipped, rotated or both
            for mtSeg in range(numSegments):
    #               index into 'smplsOf1S'
                segStartPt = np.where(time1SFromAccTrim >= qcData['mTimes'][mtSeg])[0][0]
                segEndPt = np.where(time1SFromAccTrim < qcData['mTimes'][mtSeg+1])[0][-1]
    #               index to start of seg | seconds
                qcData['elapsedSecs'][mtSeg,0] = segStartPt
    #               index seg span | 'smplsOf1S'
                segPts = np.arange(segStartPt,segEndPt+1)
    #               HELP
    #               ====
    #               walkMarkerX = np.full((len(smplsOf1S),1),np.nan)
    #               walkMarkerY = np.full((len(smplsOf1S),1),np.nan)
    #               sitMarkerZ = np.full((len(smplsOf1S),1),np.nan)
    #               ====
    #               compute mean | exclude NaN            
                nanMeanX = np.nanmean(walkMarkerX[segPts,0])
                nanMeanZ = np.nanmean(sitMarkerZ[segPts,0])
    #               get number of non-NaN values
                notNanXNum = np.sum(~np.isnan(walkMarkerX[segPts,0]))
                notNanZNum = np.sum(~np.isnan(sitMarkerZ[segPts,0]))
    #               calculate seg worn seconds | get bool
                isWornSegment = np.sum(~notWornLogic[segPts]) > minWornT
    #               save value of 'isWornSegment'
                qcData['isWornSeg'][mtSeg,0] = isWornSegment
    
    ## ->           worn seg case
                if isWornSegment:
    #                  check if 'nanMeanX' is a number | not np.nan
                    if not np.isnan(nanMeanX):
#   #                       DIFFERENCE
    #                       ==========
    #                       the logic on Matlab seems to not align with expected values
    #                       for 1g of acceleration
    #                       ==========
    #                       rotation check
                        if abs(nanMeanX-1) < 0.4:
                            qcData['xFlipValue'][mtSeg,0] = 1 #not-rot
                        elif abs(nanMeanX) < 0.6:
                            qcData['xFlipValue'][mtSeg,0] = -1 #rot
#   #                       ===END
    #                   number of seconds used to make the calculation                
                        xFlipCalcLength[mtSeg,0] = notNanXNum
    #                   flip check
                    if not np.isnan(nanMeanZ) and notNanZNum > minSitT:
                        if abs(nanMeanZ-1) < 0.4:
                            qcData['zFlipValue'][mtSeg,0] = 1 #not-flip
                        elif abs(nanMeanZ) < 0.6:
                            qcData['zFlipValue'][mtSeg,0] = -1 #flip
                        zFlipCalcLength[mtSeg,0] = notNanZNum
        
    #           index to end of last seg | 'smplsOf1S'
            qcData['elapsedSecs'][-1,0] = segEndPt
        
    ## ->       calculate orientaton during a worn period
    #           RLE to find each worn period
            iWSegValues, iWSegCounts = rle.encode(qcData['isWornSeg'][:,0])
            iWSegValues = np.array(iWSegValues, dtype=bool)
            iWSegCounts = np.array(iWSegCounts, dtype=int)
            iWSegStarts, iWSegEnds = RLEIndeces(iWSegValues,iWSegCounts)
    #           index into 'iWSegValues'
            wornSegments = np.where(iWSegValues)[0]
            for wSeg in range(len(wornSegments)):
    #               index into 'numSegments'
                wSegStart = iWSegStarts[wornSegments[wSeg]]
                wSegEnd = iWSegEnds[wornSegments[wSeg]]
                wSegNdx = range(wSegStart, wSegEnd+1)
                wSegNdxArr = np.array(wSegNdx)
        #           qcData['xFlipValue'] = np.zeros((numSegments,1))
        #           qcData['zFlipValue'] = np.zeros((numSegments,1))
        #           qcData['xFlipCalc'] = np.zeros((numSegments,1))
        #           qcData['zFlipCalc'] = np.zeros((numSegments,1))
    #               slice saved data | save temp val
                xFlipVTmp = qcData['xFlipValue'][wSegNdx,0]
                xFlipCaLeTmp = xFlipCalcLength[wSegNdx,0]
                zFlipVTmp = qcData['zFlipValue'][wSegNdx,0]
                zFlipCaLeTmp = zFlipCalcLength[wSegNdx,0]
    #               slice range 'wSegNdx' from 'mTimes' | add end value
                wSegTimes = qcData['mTimes'][np.concatenate([wSegNdxArr, wSegNdxArr+1])]
    
    ## ->           calculate worn seg rot state
    #               get non-zero bool for 'xFlipVTmp'
                xFlipVTmpNonZ = (xFlipVTmp != 0)
                dtSegStart = MatlabDatenumToDate(wSegTimes[0]).strftime('%b.%d, %H:%M')
                dtSegEnd = MatlabDatenumToDate(wSegTimes[-1]).strftime('%b.%d, %H:%M')
                if np.any(xFlipVTmpNonZ):
    #                   repeat the values of 'xFlipVTmp' according to 'xFlipCaLeTmp'
    #                   get most common value | assign to 'xFlipCalc' for 'wSegNdx'
                    qcData['xFlipCalc'][wSegNdx,0] = stats.mode(np.repeat(xFlipVTmp[xFlipVTmpNonZ],xFlipCaLeTmp[xFlipVTmpNonZ]), keepdims=False).mode
                    if np.sum(xFlipCaLeTmp[xFlipVTmpNonZ]) < minWalkT*(wSegTimes[-1] - wSegTimes[0]):
                        warnings.append(f'Rotation detection between {dtSegStart} and {dtSegEnd} uncertain')
                else:
                    qcData['xFlipCalc'][wSegNdx,0] = defRotation
                    warnings.append(f'Rotation detection between {dtSegStart} and {dtSegEnd} unsuccessful. Assuming default.')
    
    ## ->           calculate worn seg flip state
    #               get non-zero bool for 'zFlipVTmp'
                zFlipVTmpNonZ = (zFlipVTmp != 0)
                if np.any(zFlipVTmpNonZ):
                    qcData['zFlipCalc'][wSegNdx,0] = stats.mode(np.repeat(zFlipVTmp[zFlipVTmpNonZ],zFlipCaLeTmp[zFlipVTmpNonZ]), keepdims=False).mode
                    if np.sum(zFlipCaLeTmp[zFlipVTmpNonZ]) < minSitTPD*(wSegTimes[-1] - wSegTimes[0]):
                        warnings.append(f'Flip detection between {dtSegStart} and {dtSegEnd} uncertain')
                else:
                    qcData['zFlipCalc'][wSegNdx,0] = defFlip
                    warnings.append(f'Flip detection between {dtSegStart} and {dtSegEnd} unsuccessful. Assuming default.')

    ## ->       transform the raw data | use flip-rot correction
            for mtSeg in range(numSegments):
                numSecs = qcData['elapsedSecs'][mtSeg+1,0] - qcData['elapsedSecs'][mtSeg,0]
                if qcData['isWornSeg'][mtSeg,0]:
                    qcData['wornTime'] = qcData['wornTime'] + numSecs
                    qcData['rotTime'] = qcData['rotTime'] + numSecs * (qcData['xFlipCalc'][mtSeg,0] == -1)
                    qcData['flipTime'] = qcData['flipTime'] + numSecs * (qcData['zFlipCalc'][mtSeg,0] == -1)
    #               calculate orientation | use XFlipCalc and ZFlipCalc
                if exMode.lower() == 'force':
    #                  rot and flip case
                    if qcData['xFlipCalc'][mtSeg,0] == -1 and qcData['zFlipCalc'][mtSeg,0] == -1:
                        oType = 4 #rot + flip
                    elif qcData['xFlipCalc'][mtSeg,0] == -1 and qcData['zFlipCalc'][mtSeg,0] == 1:
                        oType = 3 #rot
                    elif qcData['xFlipCalc'][mtSeg,0] == 1 and qcData['zFlipCalc'][mtSeg,0] == -1:
                        oType = 2 #flip
                    else:
                        oType = 1
    #                   get 'Acc' indices for 4hr seg
                    mtSegAccI = range(smplsOf1S[qcData['elapsedSecs'][mtSeg,0]],smplsOf1S[qcData['elapsedSecs'][mtSeg+1,0]])
    #                   calculate orientation change | reassign to 'Acc'
                    Acc = ChangeAxes(Acc, devType, oType, mtSegAccI)
        
    ## ->   final stage of QC module
    #       optioanlly calculate orientation | entire Acc
        if exMode.lower() != 'force':
            if Settings[0]["Rotated"] and Settings[0]["Flipped"]:
                oType = 4
            elif Settings[0]["Rotated"] and not Settings[0]["Flipped"]:
                oType = 3
            elif not Settings[0]["Rotated"] and Settings[0]["Flipped"]:
                oType = 2
            else:
                oType = 1
            Acc = ChangeAxes(Acc, devType, oType)     
        status = 'OK'

## ->   exceptions handler
    except Exception as e:
        if status.lower() != 'ok' and exMode.lower() == 'force':
            if Settings[0]["Rotated"] and Settings[0]["Flipped"]:
                oType = 4
            elif Settings[0]["Rotated"] and not Settings[0]["Flipped"]:
                oType = 3
            elif not Settings[0]["Rotated"] and Settings[0]["Flipped"]:
                oType = 2
            else:
                oType = 1
            Acc = ChangeAxes(Acc, devType, oType)
        status = 'QC Module crashed'
        warnings.append(f'Error: {str(e)}')
        warnings.append('TRACEBACK')
        tBack = e.__traceback__
        while tBack is not None:
            frame = tBack.tb_frame
            warnings.append(f'_ERR: _function name: {frame.f_code.co_name}\n'
                            f'_ERR: _line number: {tBack.tb_lineno}\n')
            tBack=tBack.tb_next

## ->   returns
    return qcData, Acc, meanTEMP, status, warnings
                
