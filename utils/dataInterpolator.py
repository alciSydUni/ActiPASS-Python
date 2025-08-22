import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def dataInterpolator(data: pd.DataFrame, Fs: int) -> pd.DataFrame:
    interpolatedData = pd.DataFrame()
    
    #data: DataFrame provided with its own sample frequency based on what was read from 'readGenericCSV.py'
    #Fs: sample frequency provided for interpolation
    
    #set the mtStart and mtEnd time floats to exact seconds
    mtStart = round(data['time'].iloc[0]*86400)/86400
    mtEnd = round(data['time'].iloc[-1]*86400)/86400
    if mtStart == mtEnd:
        raise ValueError("Time range too short, less than 1 second. Please make sure the sample is larger.")

    #use mtStart and mtEnd and sample number -> 't' array of float values
    t = np.linspace(mtStart, mtEnd, int((mtEnd - mtStart)*86400*Fs)+1)
    
    #add time array to DataFrame
    interpolatedData['time'] = t
    
    #interpolate each column
    for col in ['X', 'Y', 'Z']:
        interpltObj = PchipInterpolator(data['time'], data[col], extrapolate=False)
        interpolatedData[col] = interpltObj(interpolatedData['time'])
        interpolatedData[col] = np.nan_to_num(interpolatedData[col],nan=0.0)

    return interpolatedData



