from datetime import datetime, timedelta
from typing import Tuple, List, Union
import pandas as pd
import numpy as np
from library import HelperFunctions

def readGenericCSV(PATH:str) -> Tuple[pd.DataFrame,float,float,str]:
    
    ## -> set working variables
    Data = pd.DataFrame()
    SF: float = np.nan
    deviceID: float  = np.nan
    devType: str = ""

    ## - > try to read file metadata/data
    try:
        headLs = [""] * 10
        headerFound = False
        numLs = 0
        with open(PATH) as file:
            while numLs < 10:
                headLs[numLs] = file.readline().strip()
                if not headLs[numLs]:
                    headLs[numLs] = ""
                    break
                if set(headLs[numLs].replace(" ", "").lower().split(",")) == {"x", "y", "z"}:
                    headerFound = True
                    file.seek(0)
                    lines = file.readlines()
                    break
                numLs += 1
        #no x,y,z header means no data
        if not headerFound:
            raise ValueError("Unrecognized generic CSV format")
        #extract devType if it exists, if not set to 'Generic' -> str
        devType = str(valueExtract(headLs,"devtype",True) or "Generic")
        #extract deviceID, SF -> float
        deviceID = float(valueExtract(headLs, "id",False))
        SF = float(valueExtract(headLs, "sf",False))
        #extract startT string -> float
        startT_str = str(valueExtract(headLs,"start", True))
        if startT_str != "": 
            startT_datetime = datetime.strptime(startT_str,"%Y%m%dT%H%M%S.%f")
            #startT = matlabDatenum(startT_datetime)
            startT = HelperFunctions.matlabDatenum(startT_datetime)

        else:
            raise ValueError("Missing START timestamp in header.")
        #read file contents -> determine start of x,y,z -> read all values for x,y,z
        for i, line in enumerate(lines):
            if set(line.replace(" ","").lower().strip().split(",")) == {"x","y","z"}:
                start_index = i
                break
        else:
            raise ValueError("No x,y,z header found")
        #create X, Y, Z DataFrame
        Data = pd.read_csv(PATH, skiprows=start_index+1, names= ["X", "Y", "Z"], dtype={"X": float, "Y": float, "Z": float}, na_values=["", "NaN"])
        #add timestamps to DataFrame
        if not np.isnan(SF) and SF > 0:
            singleSampleDelta = 1 / SF / 86400 #sample fraction in days
            timeArray = [startT + i * singleSampleDelta for i in range(Data.shape[0])]
            Data['time'] = timeArray
        else:
            raise ValueError("SF is NaN or smaller than 1")
    except Exception as e:
        print(f'Error: {e}')

    return Data, SF, deviceID, devType
 

def valueExtract(headLs: List[str], key: str, isStringBool: bool):
    for line in headLs:
        lineParts = line.strip().split("=",1)
        if len(lineParts) == 2 and lineParts[0].strip().lower() == key.lower():
            value = lineParts[1].strip()
            if isStringBool:
                return value
            try:
                return float(value)
            except ValueError:
                return float("nan")
    return "" if isStringBool else float("nan")

## ->   everything below moved to library.HelerFunctions.py
"""
def matlabDatenum (date: datetime) -> float:
    return (date - datetime(1,1,1) + timedelta(days= 366)).total_seconds() / 86400

def matlabDatenumArr (dates: Union[List[datetime], np.ndarray]) -> np.ndarray:
    yearOne = datetime(1,1,1)
    delta = timedelta(days=366)
    return np.array([(d - yearOne + delta).total_seconds()/86400 for d in dates]) #(N,)
"""
    
