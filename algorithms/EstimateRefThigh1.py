import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray
import pandas as pd
import ActivityDetect

def EstimateRefThigh1(Acc: pd.DataFrame, vThigh: np.ndarray, VRefThigh: np.ndarray, VRefThighDef: np.ndarray, SF: int, ParamsAP: dict) -> NDArray[np.float64]:
    actDetect, _, _, _ = ActivityDetect.ActivityDetect(Acc, SF, Acc['time'], VRefThigh, ParamsAP)
    walkAvgVRad = np.pi*(11/180)
    vMeanZ = np.mean(vThigh[:,1].reshape((SF, len(actDetect)),order='F'), axis=0)
    vMeanY = np.mean(vThigh[:,2].reshape((SF, len(actDetect)), order='F'), axis=0)
    vMedZ = np.median(vMeanZ[(actDetect == 5)]) - walkAvgVRad
    vMedY = np.median(vMeanY[(actDetect == 5)])
    vRefThigh = np.asarray([np.acos(np.cos(vMedZ)*np.cos(vMedY)), vMedZ, vMedY],dtype=np.float64)
    if np.isnan(vMedZ) or np.sum(actDetect == 5)<30:
        vRefThigh = np.asarray(VRefThighDef,dtype=np.float64)
    return vRefThigh

