from readGenericCSV import readGenericCSV
import pandas as pd

def readGenericCSV_TF():
    Data, SF, deviceID, devType = readGenericCSV('/Users/alpopp/Documents/Work/Al/2025 USYD/VS/ActiPASS - Python/utils/readGenericCSV_assets/genericCSV.csv')
    print("Data:")
    pd.set_option("display.precision", 15)
    print(Data)
    print()
    print(f"SF: {SF}")
    print()
    print(f"deviceID: {deviceID}")
    print()
    print(f"devType: {devType}")

if __name__ == "__main__":
    readGenericCSV_TF()
