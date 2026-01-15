import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray
import pandas as pd

def EstimateRefThigh1(Acc: pd.DataFrame, VThigh: pd.DataFrame, VRefThigh: np.ndarray, VRefThighDef: np.ndarray, SF: int, ParamsAP: dict) -> pd.DataFrame:
        actDetect = 