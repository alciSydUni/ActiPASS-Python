import LoadSettings
import json

def LoadSettings_TF():
    ## -> set absolute path for simplicity
    ActiPASSConfig_path = '/Users/alpopp/Documents/Work/Al/2025 USYD/VS/ActiPASS - Python/library/LoadSettings_assets/ActiPASSConfig_TJSON.json'
    ParamsAP_path = '/Users/alpopp/Documents/Work/Al/2025 USYD/VS/ActiPASS - Python/library/LoadSettings_assets/ParamsAP_TJSON.json'

    ActiPASSConfig_path2 = '/Users/alpopp/Documents/Work/Al/2025 USYD/VS/ActiPASS - Python/library/LoadSettings_assets/test_path/ActiPASSConfig_TJSON.json'
    ParamsAP_path2 =  '/Users/alpopp/Documents/Work/Al/2025 USYD/VS/ActiPASS - Python/library/LoadSettings_assets/test_path/ParamsAP_TJSON.json'

    ## -> load parameters and print
    try:
        Settings, ParamsAP = LoadSettings.LoadSettings(ActiPASSConfig_path,ParamsAP_path)
        print()
        print("Settings loaded successfully:", Settings)
        print()
        print("ParamsAP loaded successfully:", ParamsAP)
    except FileNotFoundError as fnf:
        print (f"File not found error: {fnf}")
    except json.JSONDecodeError as jsonError:
        print(f"JSON decode error: {jsonError}")
    except Exception as e:
        print(f"General error: {e}")

LoadSettings_TF()
