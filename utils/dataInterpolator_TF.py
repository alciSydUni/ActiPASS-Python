import numpy as np
import pandas as pd
from readGenericCSV import readGenericCSV
from dataInterpolator import dataInterpolator
import os
import inspect
import matplotlib.pyplot as plt

#use genericCSV.csv with 'readGenericCSV.py' to get 'Data' DataFrame return
#call 'dataInterpolator.py' with new 'Fs' to get interpolatedData DataFrame return

Data: pd.DataFrame
interpolatedData: pd.DataFrame
SF: float
deviceID: float
devType: str

#read CSV and return values
Data, SF, deviceID, devType= readGenericCSV('/Users/alpopp/Documents/Work/Al/2025 USYD/VS/ActiPASS - Python/utils/readGenericCSV_assets/genericCSV.csv')
#print(Data)
print(Data.assign(time=Data['time'].map(lambda x: f"{x: .15f}")))


#call 'dataInterpolator.py'
interpolatedData = dataInterpolator(Data,50)
#print(interpolatedData)
print(interpolatedData.assign(time=interpolatedData['time'].map(lambda x: f"{x: .15f}")))


try:
    call_file = os.path.basename(inspect.stack()[0].filename) 
    if call_file == 'dataInterpolator_TF.py':
        for col in ['X', 'Y', 'Z']:
            plt.figure(num="Data vs Interpolated Data",figsize=(20,5))
            plt.plot(Data['time']*86400, Data[col],'o-', label=col+" Data - 10Hz")
            plt.plot(interpolatedData['time']*86400, interpolatedData[col], 'x-', label=col+" Interpolated - 50Hz")
            plt.xlabel('Time (s)')
            plt.ylabel('Values')
            plt.title('Data vs Interpolated Data')
            plt.legend()
            plt.tight_layout()
            plt.show()

except Exception as e:
    print(f"Exception {e}")
