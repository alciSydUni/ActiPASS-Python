import numpy as np
import pandas as pd

def NotWorn(vThigh: np.ndarray,accDF: pd.DataFrame,SF: int) -> np.ndarray:

## ->   resample to seconds
    vThigh = vThigh[0::SF,:]

## ->   compute standard deviation
    stdMov = accDF.rolling(window=SF*2,min_periods=SF*2).std(ddof=0).to_numpy()
    stdMov1S = stdMov[1::SF,:]
    stdMov1SMean = np.mean(stdMov1S,axis=1)
    stdMov1SSum = np.sum(stdMov1S,axis=1)

## ->   resolve non-worn periods
    wornNW = np.diff(np.concatenate(([False],stdMov1SMean[0:-1]<0.01,[False])))
    NW = np.where(wornNW==1)[0]
    worn = np.where(wornNW==-1)[0]
    if NW.size:
        nwPeriods = worn - NW
        nwStarts = NW[nwPeriods>600]
        nwEnds = worn[nwPeriods>600]

## ->       filter non-worn periods - some activity before OR >90min
        keepNW = np.full_like(nwStarts,False,dtype=bool)
        for i in range(len(nwStarts)):
            keepNW[i]=(np.max(stdMov1SSum[np.arange(max(nwStarts[i]-15,0),max(nwStarts[i]-11,4)+1)])>0.5 or
                        (nwEnds[i]-nwStarts[i])>5400)
        nwStarts = nwStarts[keepNW]
        nwEnds=nwEnds[keepNW]

## ->       filter non-worn periods - activity of <1min between NW
        shortWorn = (nwStarts[1:]-nwEnds[:-1])<60
        nwEnds = nwEnds[np.concatenate((~shortWorn,[True]))]
        nwStarts = nwStarts[np.concatenate(([True], ~shortWorn))]

## ->       filter non-worn periods - thighV for lie
        nwReturn = np.zeros((len(vThigh),),dtype=bool)
        for i in range(len(nwStarts)):
            vMean = (180/np.pi)*np.mean(vThigh[nwStarts[i]:nwEnds[i]+1,:],axis=0) #vThigh is indexed at sample level | nwStarts/Ends is indexed at 1s level
            if (nwEnds[i]-nwStarts[i]>5400 or
                np.all(np.abs(vMean-np.asarray([90,90,0]))<5) or
                np.all(np.abs(vMean-np.asarray([90,-90,0]))<5)
                ):
                nwReturn[nwStarts[i]:nwEnds[i]+1] = True #again, mismatched indexing between samples and seconds on Matlab side

## ->       filter non-worn periods - make NaN in stdMov1SMean as non-worn
        nwReturn[np.isnan(stdMov1SMean)] = True #again, mismatched indexing between samples and seconds on Matlab side
        
## ->       return value
        return nwReturn
    else:
        return np.zeros(len(vThigh), dtype=bool)

        
            