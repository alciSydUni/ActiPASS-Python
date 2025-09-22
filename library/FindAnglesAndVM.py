import numpy as np
import numpy.typing as npt
import scipy.signal as scisig
from typing import Tuple, List, Optional
import pandas as pd

## -> numpy Array

def FindAnglesAndVM(Acc,SF,Fc):
    ## -> informative type annotations --> 64bit coefficients
    Blp: npt.NDArray[np.float64] 
    Alp: npt.NDArray[np.float64]
    ## -> Butterworth filter
    Blp, Alp = scisig.butter(N=6, Wn=Fc, btype = 'low', output = 'ba', fs =SF)
    ## -> apply filter -> use axis=0 for time series
    AccFilt = scisig.lfilter(Blp, Alp, Acc, axis=0)
    SVM=np.sqrt(np.sum(AccFilt ** 2, axis=1))
    normAcc = AccFilt/SVM[:,np.newaxis]
    ## -> angle between normalized 3D vector and X-axis unit vector [1,0,0]
    Inc = np.arccos(normAcc[:,0])
    ## -> angle between horizontal Z and downward roll around Y 
    #equivalent to the strength of the 3D vector in the Z direction
    # !! why is forward lean +ve Z = to negative angle through -ve sign?
    U = -np.arcsin(normAcc[:,2])
    ## -> angle between horizontal Y and X-Z plane 
    #+ve acceleration = dip in front facing Y-axis
    #resulting in -ve angle, which is flipped to +ve
    V = np.column_stack((Inc, U, -np.arcsin(normAcc[:,1])))
    return V, AccFilt, SVM, normAcc

## -> pandas DataFrame

def FindAnglesAndVM_DF(
    Acc: pd.DataFrame,
    SF: float,
    Fc: float,
    smplsOf1S: Optional[List[int]] = None #list: index values for Acc - Fs apart - starting from 0
) -> tuple[npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64]]:

    ## -> convert DataFrame to numpy array
    AccNumpy = Acc[['X','Y','Z']].to_numpy(dtype=np.float64)
    ## -> Butterworth filter
    Blp: npt.NDArray[np.float64]
    Alp: npt.NDArray[np.float64]
    Blp, Alp = scisig.butter(N=6, Wn=Fc, btype = 'low', output = 'ba', fs =SF) #type: ignore
    ## -> apply filter -> use axis=0 for time series
    accFilt: npt.NDArray[np.float64] = scisig.lfilter(Blp, Alp, AccNumpy, axis=0) #type: ignore
    if smplsOf1S is not None:
        accFilt = accFilt[smplsOf1S,:] 
    SVM=np.sqrt(np.sum(accFilt ** 2, axis=1))
    normAcc = accFilt/SVM[:,np.newaxis]
    ## -> angle between normalized 3D vector and X-axis unit vector [1,0,0]
    Inc = np.arccos(normAcc[:,0])
    ## -> angle between horizontal Z and downward roll around Y 
    #equivalent to the strength of the 3D vector in the Z direction
    # !! why is forward lean +ve Z = to negative angle through -ve sign?
    U = -np.arcsin(normAcc[:,2])
    ## -> angle between horizontal Y and X-Z plane 
    #+ve acceleration = dip in front facing Y-axis
    #resulting in -ve angle, which is flipped to +ve
    V = np.column_stack((Inc, U, -np.arcsin(normAcc[:,1])))
    return V, accFilt, SVM, normAcc
    