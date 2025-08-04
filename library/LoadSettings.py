import os
import json
from pathlib import Path
from typing import Optional

def LoadSettings(ActiPASSConfig: str, ParamsConfig: str) -> tuple[dict,dict]:
    Settings: dict = {}
    ParamsAP: dict = {}
    
    ## -> ParamsConfig -> check IF folder exits, if not create it
    configDir = os.path.dirname(ParamsConfig)
    if configDir !="" and not os.path.isdir(configDir):
        os.makedirs(configDir, exist_ok=True)

    ## -> ParamsConfig -> check IF file exists and load it -> ParamsAP
    if os.path.isfile(ParamsConfig):
        try:
            with open(ParamsConfig, mode="r") as file:
                ParamsAP = json.load(file)
        except Exception as e:
            print(f"Failed to load parameters from ParamsConfig file: {e}")
    else:
        print(f"There is no ParamsConfig file along the provided path: {ParamsConfig}")

    ## -> ParamsConfig -> ParamsAP -> define paramter names
    paramNms=["Fs","Fc","VrefThighMin","VrefThighMax","VrefThighDef","VrefTrunkMin","VrefTrunkMax","VrefBackDef",
    "VrefChestDef","VrefCalfMin","VrefCalfMax","VrefCalfDef","Bout_cycle","Bout_lie","Bout_move","Bout_row",
    "Bout_run","Bout_sit","Bout_stair","Bout_stand","Bout_walk","Threshold_sitstand","Threshold_staircycle",
    "Threshold_standmove","Threshold_walkrun","Threshold_slowfastwalk","Threshold_veryfastwalk","Threshold_kneel",
    "Threshold_HF_energy","Threshold_SDMax_Acc","MET_SI","MET_LieStill","MET_Lie","MET_Sit","MET_Stand",
    "MET_Move","Wlk_Low_MET","Wlk_Fast_MET","Wlk_VFast_MET","MET_Running","MET_Stairs","MET_Cycle","MET_Other",
    "MET_INT_Slp","MET_INT_SED","MET_INT_LPA","MET_INT_LPA_Amb","MET_INT_MPA","MET_INT_VPA","VarsBout"]
    
    #TEST -> test if default is loaded
    ## -> ParamsAP is empty OR not all parameters exist -> use default settings
    if not ParamsAP or not all(name in ParamsAP for name in paramNms):
        #resampling and filtering of raw accelerometer data
        ParamsAP["Fs"]=25 #the resample frequency for raw accelerometer data
        ParamsAP["Fc"]=2; #the cutoff frequency for angle and vector-magnitude finding (primary filter)
        #Acti4 algorithm related settings
        ParamsAP["Bout_cycle"] = 15
        ParamsAP["Bout_lie"] = 5
        ParamsAP["Bout_move"] = 2
        ParamsAP["Bout_row"] = 15
        ParamsAP["Bout_run"] = 2
        ParamsAP["Bout_sit"] = 5
        ParamsAP["Bout_stair"] = 5
        ParamsAP["Bout_stand"] = 2
        ParamsAP["Bout_walk"] = 2
        ParamsAP["Threshold_sitstand"] = 45
        ParamsAP["Threshold_staircycle"] = 40
        ParamsAP["Threshold_standmove"] = 0.1
        ParamsAP["Threshold_walkrun"] = 0.72
        ParamsAP["Threshold_kneel"]=75
        #cadance cutoffs for slow/fast and fast/very fast walking (used both in batch process and stats generation)
        ParamsAP["Threshold_slowfastwalk"] = 100
        ParamsAP["Threshold_veryfastwalk"] = 135
        #transport algorithm related settings
        ParamsAP["Threshold_HF_energy"] = 0.1
        ParamsAP["Threshold_SDMax_Acc"] = 0.015
        #stat generation: behaviours/int-classes for which bouts/percentiles are generated
        ParamsAP["VarsBout"]=["SitLie","Upright","INT34"]
        #stat generation MET cutoffs for behaviours
        ParamsAP["MET_SI"]=0.90
        ParamsAP["MET_LieStill"]=0.95
        ParamsAP["MET_Lie"]=1.0
        ParamsAP["MET_Sit"]=1.3
        ParamsAP["MET_Stand"]=1.55 # 2022-07-05 standing falls into light physical activity class (changed from 1.4)
        ParamsAP["MET_Move"]=2.0
        ParamsAP["Wlk_Low_MET"]=2
        ParamsAP["Wlk_Fast_MET"]=4
        ParamsAP["Wlk_VFast_MET"]=7
        ParamsAP["MET_Running"]=10
        ParamsAP["MET_Stairs"]=8
        ParamsAP["MET_Cycle"]=7
        ParamsAP["MET_Other"]=2 # "Other" with no periodicity falls into light physical activity
        #stat generation MET cutoffs for intensity classes
        ParamsAP["MET_INT_Slp"]=0.0
        ParamsAP["MET_INT_SED"]=0.95 #lieStill belongs to sedentary
        ParamsAP["MET_INT_LPA"]=1.5
        ParamsAP["MET_INT_LPA_Amb"]=1.6 #introduce another called LPA_ambulatory to seperate standing from other LPA activities
        ParamsAP["MET_INT_MPA"]=3.0
        ParamsAP["MET_INT_VPA"]=6.0
        #individual reference position values for thigh min, max and defaults
        ParamsAP["VrefThighMin"] = [0,-32,-15]
        ParamsAP["VrefThighMax"] = [32,0,15]
        ParamsAP["VrefThighDef"] = [16,-16,0]
        #individual reference position values for trunk min, max and defaults
        ParamsAP["VrefTrunkMin"] = [0,5,-15]
        ParamsAP["VrefTrunkMax"] = [53,50,15]
        ParamsAP["VrefBackDef"] = [27,27,0] # for back accelerometer
        ParamsAP["VrefChestDef"] = [10,10,0] # for chest accelerometer
        #individual reference position values for calf min, max and defaults
        ParamsAP["VrefCalfMin"] = [0,-15,-15]
        ParamsAP["VrefCalfMax"] = [15,15,15]
        ParamsAP["VrefCalfDef"] = [0,0,0]

    ## -> ActiPASSConfig -> check IF folder exits, if not create it
    configDir = os.path.dirname(ActiPASSConfig)
    if configDir !="" and not os.path.isdir(configDir):
        os.makedirs(configDir, exist_ok=True)

    ## -> ActiPASSConfig -> check if file exist and load it -> Settings
    if os.path.isfile(ActiPASSConfig):
        try:
            with open(ActiPASSConfig, mode="r") as file:
                Settings = json.load(file)
        except Exception as e:
            print(f"Failed to load parameters from ActiPASSConfig file: {e}")
    else:
        print(f"There is no ActiPASSConfig file along the provided path: {ActiPASSConfig}")
    
    #flag for Activating ProPASS mode of ActiPASS_GUI
    ValidateRoutine(Settings,"PROPASS", None,True)
    NormalizeRoutine(Settings,"PROPASS")
    #flag for showing advanced options dialog
    ValidateRoutine(Settings,"ADVOPTIONS",None,True)
    NormalizeRoutine(Settings,"ADVOPTIONS")
    #flag for enabling experimental features
    ValidateRoutine(Settings,"LAB",None,False)
    NormalizeRoutine(Settings,"LAB")
    
    #TEST -> add in non path values to JSON to test Path adjustment
    #set file and folder paths (in order to save and load last used files/folders)
    for key in ["thighAccDir", "trunkAccDir", "calfAccDir", "diary_file", "out_folder"]:
        if (key not in Settings or not isinstance(Settings[key],str)
            or not Path(Settings[key]).exists()
        ): Settings[key] = str(Path.home())
    if ('cal_file' not in Settings or not isinstance(Settings["cal_file"],str)
            or not Path(Settings["cal_file"]).exists()
        ): Settings["cal_file"] = str(Path.home() / 'DeviceCal.csv')
    
    ## -> Set ID related settings
    #set subject-ID mode
    ValidateRoutine(Settings,"IDMODE",["start","end","activpal","full-filename"],"full-filename")
    #set number of ID digits
    ValidateRoutine(Settings,"IDLENGTH",None,5)
    
    ## -> Set device calibration settings
    #flag for autocalibrating data
    ValidateRoutine(Settings,"CALMETHOD",["off","auto","file"],"auto")
    #flag for appending autocalibrating data to calibration file
    ValidateRoutine(Settings,"ADDAUTOCAL",None,True)
    NormalizeRoutine(Settings,"ADDAUTOCAL")

    ## -> Settings related to thigh Acc and Flips/Rots/NWTrim module
    #flag for auto flip/rotation corrections
    ValidateRoutine(Settings,"FLIPROTATIONS",["warn","force"],"force")
    # !! flag for rotated orientation, only used when FLIPROTATIONS=false
    ValidateRoutine(Settings,"Rotated",None,False)
    NormalizeRoutine(Settings,"Rotated")
    # !! flag for flipped' orientation, only used when FLIPROTATIONS=false  
    ValidateRoutine(Settings,"Flipped",None,False)
    NormalizeRoutine(Settings,"Flipped")
    #flag for reference position finding method
    ValidateRoutine(Settings,"REFPOSTHIGH",["default","auto1","auto2","diary"],"auto1")

    ## -> Set NW related settings
    #flag for automatic cropping NW at begining or end
    ValidateRoutine(Settings,"TRIMMODE",["off","force","nodiary"],"nodiary")
    #min length of short NW periods within active periods to ignore in hrs
    ValidateRoutine(Settings,"NWSHORTLIM",None,3)
    #buffer around NW cropping (used in Flips/Rots/NWTrim module)
    ValidateRoutine(Settings,"NWTRIMBUF",None,1)
    #min length of consecutive wear period to consider for trimming data
    ValidateRoutine(Settings,"NWTRIMACTLIM",None,24)
    #NW correction using bedtime based on lying
    ValidateRoutine(Settings,"NWCORRECTION",["lying","fixed","extra"],"lying")

    ## -> Set cadence, seated-transport lying, bedtime and sleep related settings
    #flag for cadence detection algorithm
    ValidateRoutine(Settings,"CADALG",["FFT","Wavelet1","Wavelet2"],"FFT")
    #enable or disable seated transport detection
    ValidateRoutine(Settings,"TRANSPORT",["on","off"],"off")
    #different lying algorithms to find lying periods
    ValidateRoutine(Settings,"LIEALG",["off","auto","diary","algA","algB","trunk"],"auto")
    #sleep algorithm: currently only Skotte
    ValidateRoutine(Settings,"SLEEPALG",["off","In-Bed","InOut-Bed","diary"],"In-Bed")
    #flag for considering no-sleep-interval for final QC_Status - enable or disable
    ValidateRoutine(Settings,"CheckSlpInt",["on","off"],"on")
    #flag for bedtime algorithm
    ValidateRoutine(Settings,"BEDTIME",["off","auto1","auto2","diary"],"auto2")
    #bedtime algorithm max-active-time threshold
    ValidateRoutine(Settings,"BDMAXAKTT",None,20,rangeCheck=(1,60))
    #bedtime algorithm min-lie-time threshold
    ValidateRoutine(Settings,"BDMINLIET",None,180,rangeCheck=(60,720))
    #bedtime algorithm very-long-sit threshold
    ValidateRoutine(Settings,"BDVLONGSIT",None,240,rangeCheck=(120,720))
    #flag for saving the 1Hz activity and steps data
    ValidateRoutine(Settings,"EXTERNFUN",None,False)
    NormalizeRoutine(Settings,"EXTERNFUN")
    
    ## -> Set trunk accelerometer settings
    #set trunk accelerometer position and also enable it
    ValidateRoutine(Settings,"TRUNKPOS",["off","back","front"],"off")
    #trunk accelerometer filename suffix
    ValidateRoutine(Settings,"TRUNKSUFFIX",None,"")
    #flag for flipped (inside-out) orientation
    ValidateRoutine(Settings,"TRUNKFLIP",None,False)
    NormalizeRoutine(Settings,"TRUNKFLIP")
    #flag for rotated (upside-down) orientation
    ValidateRoutine(Settings,"TRUNKROT",None,False)
    NormalizeRoutine(Settings,"TRUNKROT")
    #flag for force-synchronization with thigh
    ValidateRoutine(Settings,"FORCESYNC",None,True)
    NormalizeRoutine(Settings,"FORCESYNC")
    #trunk accelerometer filename prefix
    ValidateRoutine(Settings,"TRUNKPREFIX",None,"")
    #keep trunk accelerometer NW as NW in the final result
    ValidateRoutine(Settings,"KEEPTRUNKNW", None,True)
    NormalizeRoutine(Settings,"KEEPTRUNKNW")
    #flag for reference position finding method
    ValidateRoutine(Settings,"REFPOSTRNK",["default","auto1","diary"],"auto1")
    #save trunk angle data for further processing
    ValidateRoutine(Settings,"SAVETRNKD",None,False)
    NormalizeRoutine(Settings,"SAVETRNKD")
    #attempt trunk orientation correction
    ValidateRoutine(Settings,"FLIPROTTRNK",None,False)
    NormalizeRoutine(Settings,"FLIPROTTRNK")

    ## -> Set calf accelerometer settings
    #calf accelerometer position and also enable it
    ValidateRoutine(Settings,"CALFPOS", ["off","on"],"off")
    #calf accelerometer filename suffix
    ValidateRoutine(Settings,"CALFSUFFIX",None,"")
    #flag for flipped (inside-out) orientation
    ValidateRoutine(Settings,"CALFFLIP", None,False)
    NormalizeRoutine(Settings,"CALFFLIP")
    #flag for rotated (upside-down) orientation
    ValidateRoutine(Settings,"CALFROT",None,False)
    NormalizeRoutine(Settings,"CALFROT")
    #flag for force-synchronization with thigh
    ValidateRoutine(Settings,"FORCESYNCCALF",None,True)
    NormalizeRoutine(Settings,"FORCESYNCCALF")
    #calf accelerometer filename prefix
    ValidateRoutine(Settings,"CALFPREFIX",None,"")
    #keep calf accelerometer NW as NW in the final result
    ValidateRoutine(Settings,"KEEPCALFNW",None,True)
    NormalizeRoutine(Settings,"KEEPCALFNW")
    #flag for reference position finding method
    ValidateRoutine(Settings,"REFPOSCALF",["default","diary"],"default")
    #attempt calf orientation correction
    ValidateRoutine(Settings,"FLIPROTCALF",None,False)
    NormalizeRoutine(Settings,"FLIPROTCALF")
    #max kneeling minutes per day before flagging
    ValidateRoutine(Settings,"maxKneelDur",None,60)

    ## -> Set stagel outputs and visualization settings
    #flag for saving the 1Hz activity and steps data
    ValidateRoutine(Settings,"SAVE1SDATA",None,True)
    NormalizeRoutine(Settings,"SAVE1SDATA")
    #flag for visualization option
    ValidateRoutine(Settings,"VISUALIZE",["off","full","QC","extra"],"QC")
    #histogram bin size
    ValidateRoutine(Settings,"histgStep",None,60)
    #minimum valid duration for consideration of day for histogram inclusion
    ValidateRoutine(Settings,"histgMinDur",None,22)
    #flag for pie charts
    ValidateRoutine(Settings,"PIECHARTS",None,True)
    NormalizeRoutine(Settings,"PIECHARTS")
    #flag for diary comments
    ValidateRoutine(Settings,"PRINTCOMMNTS",None,True)
    NormalizeRoutine(Settings,"PRINTCOMMNTS")

    ## -> Set outlier detection and stage3 stats generation settings
    #ignore bad or problematic files in stats genneration module
    ValidateRoutine(Settings,"statsIgnoreQC",["NotOK","NotOK+Check","None"],"NotOK")
    #domains for stat generation (compared against diary events)
    ValidateRoutine(Settings,"STATDOMAINS",None,"")
    #how stat domains are compared with diary events
    ValidateRoutine(Settings,"StatMtchMode",["Inclusive","Strict"],"Inclusive")
    #for stat-domain calculations, treat diary bedtime as leisure
    ValidateRoutine(Settings,"DBedAsLeis",None,True)
    NormalizeRoutine(Settings,"DBedAsLeis")
    #valid-day threshold
    ValidateRoutine(Settings,"minValidDur",None,20)
    #number of days to generate stats
    ValidateRoutine(Settings,"statNumDays",None,7)
    #how valid days are found when more measurement days than statNumDays present
    ValidateRoutine(Settings,"statSlctDays",["first valid days", "pick window: optimal work/leisure","pick days: optimal work/leisure"],"first valid days")
    #criteria for daily validity
    ValidateRoutine(Settings,"StatsVldD",["ProPASS", "only wear-time"],"ProPASS")
    #minimum walking seconds before flagging as no-walking
    ValidateRoutine(Settings,"minWlkDur",None,30)
    #maximum 'other' minutes allowed before flagging
    ValidateRoutine(Settings,"maxOtherDur",None,30)
    #max stair walking minutes per day before flagging
    ValidateRoutine(Settings,"maxStairDur",None,120)
    #enable or disable bouts generation
    ValidateRoutine(Settings,"genBouts",["on","off"],"off")
    #bout threshold value
    ValidateRoutine(Settings,"boutThresh",None,0)
    #bout break for all bouts except 1 min bout in seconds
    ValidateRoutine(Settings,"boutBreak",None,20)
    #how to calculate MET values for walking, fixed cutoffs or cadence based regression? default - 'fixed'
    ValidateRoutine(Settings,"WalkMET",["fixed","regression"],"fixed")
    #flag to calculating cadence per minute instead of per second (before slow/fast walking detection and INT classes)
    ValidateRoutine(Settings,"CADPMIN",None,False)
    NormalizeRoutine(Settings,"CADPMIN")
    #set exponential smoothing on TAI options
    ValidateRoutine(Settings,"FilterTAI",["off","TC10","TC20","TC30","TC60","TC90","TC120"],"off")
    #set stat-generation table format options
    ValidateRoutine(Settings,"TblFormat",["Daily","Events","EventsNoBreak","Daily+Events","Hourly"],"Daily")
    
    return(Settings,ParamsAP)
    
### -> helper routines

def ValidateRoutine(Settings, key, optionList: Optional[list], default, *, isNumericCheck=False, rangeCheck: Optional[tuple]=None):
    val = Settings.get(key)
    if key not in Settings or val is None or (isinstance(val,str) and val.strip() == ""):
        Settings[key] = default
        return
    if optionList is not None and str(val).strip().lower() not in [ol.strip().lower() for ol in optionList]:
        Settings[key] = default
        return
    if isNumericCheck == True and isinstance(val,(int,float)):
        Settings[key] = default
        return
    if rangeCheck is not None and not (rangeCheck[0] <= val <= rangeCheck[1]):
        Settings[key] = default
        return
    
def NormalizeRoutine(Settings, key):
    val = Settings.get(key, False)
    if str(val).strip().lower() in ["false","0",""]:
        Settings[key] = False
    else:
        Settings[key] = bool(val)

    
                



