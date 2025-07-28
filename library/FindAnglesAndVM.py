import numpy as np
import numpy.typing as npt
import scipy.signal as scisig

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
