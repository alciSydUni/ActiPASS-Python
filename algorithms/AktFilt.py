import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray
import pandas as pd
import scipy.signal
import scipy.ndimage

def AktFilt(combArr: np.ndarray,actType: str,ParamsAP: dict) -> np.ndarray:

## ->   select the correct type
    lookup = {
        "lie":(ParamsAP["Bout_lie"],1),
        "sit":(ParamsAP["Bout_sit"],2),
        "stand":(ParamsAP["Bout_stand"],3),
        "move":(ParamsAP["Bout_move"],4),
        "walk":(ParamsAP["Bout_walk"],5),
        "walkslow":(ParamsAP["Bout_walk"],5.1),
        "walkfast":(ParamsAP["Bout_walk"],5.2),
        "run":(ParamsAP["Bout_run"],6),
        "stair":(ParamsAP["Bout_stair"],7),
        "cycle":(ParamsAP["Bout_cycle"],8),
        "row":(ParamsAP["Bout_row"],9)
    }
    bout, No = lookup.get(actType, (None, None))

## ->   analyze the current actType
    combArrNew = combArr.copy()
    aktArr = np.zeros_like(combArr)
    aktArr[combArr == No] = 1
    diffAkt = np.diff(np.concatenate(([0],aktArr,[0])))
#       sAkt & eAkt index aktArr   
    sAkt = np.where(diffAkt == 1)[0]
    eAkt = np.where(diffAkt == -1)[0]-1
#       length of each bout
    bLen = eAkt - sAkt + 1
#       which bLen are too short
    shortB = np.where(bLen<bout)[0]
    startStop = np.column_stack((sAkt[shortB],eAkt[shortB]))

## ->   adjust the combArrNew to reflect removal of short bouts
    for i in range(len(startStop)):
        if i==0 and startStop[i,0]==0:
            combArrNew[startStop[i,0]:startStop[i,1]+1] = combArr[startStop[i,1]+1]
        elif i == len(startStop)-1 and startStop[i,1] == len(aktArr)-1:
            combArrNew[startStop[i,0]:startStop[i,1]+1] = combArr[startStop[i,0]-1]
        else:
            midPt = int(np.mean(startStop[i,:]))
            combArrNew[startStop[i,0]:midPt+1] = combArr[startStop[i,0]-1]
            combArrNew[midPt+1:startStop[i,1]+1] = combArr[startStop[i,1]+1]

## ->   return
    return combArrNew

