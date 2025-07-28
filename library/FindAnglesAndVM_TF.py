import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from FindAnglesAndVM import FindAnglesAndVM
## --> put signalZ in the library folder before running

def FindAnglesAndVM_TF(SF,Fc):
    ## -> sample data -> create new one
    t = np.linspace(0,10,SF*10) #10 seconds @ 25Hz = 250 time points array
    signalX=np.sin(2*np.pi*1*t) #2pirad x 1s = 1Hz sin wave
    signalY=0.5*np.sin(2*np.pi*0.5*t) #1pirad x 1s = 0.5Hz sin wave w/ half amplitude
    ## -> signalZ -> create new one
    #signalZ=0.3*np.random.randn(len(t)) #low amplitude random noise
    import os
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir,'signalZ.csv')
    ## -> signalZ.csv -> export new one
    #np.savetxt(file_path,signalZ,delimiter=',')
    ## -> signalZ.csv -> import current one
    signalZ=np.loadtxt(file_path,delimiter=',')
    ## -> signalX + signalY + signalZ -> combime col arrays
    signal=np.column_stack((signalX,signalY,signalZ))
    ## -> run FindAnglesAndVM
    V, AccFilt, SVM, normAcc = FindAnglesAndVM(signal,SF,Fc) 
    ## -> plot results
    fig = plt.figure(num='Python', figsize=(12,6))
    #fig.canvas.manager.set_window_title('Python')
    labels=['Inclination (X)','Roll (Z)', 'Pitch(Y)']
    try:
        for i in range(3):
            plt.plot(t,np.degrees(V[:,i]),label=labels[i])
    except Exception as e:
        print("Plot error:", e)
    plt.title('Angles against time')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)
    ## -> extra print values
    print("Extra print values:")
    print("Length of t:", len(t))
    print("First 10 values of t:", t[:10])
    print("Length of V:", len(V), "Shape of V:", V.shape)
    print("Length of AccFilt:", len(AccFilt), "Shape of AccFilt:", AccFilt.shape)
    print("Length of SVM:", len(SVM), "Shape of SVM:", SVM.shape)
    print("Length of normAcc:", len(normAcc), "Shape of normAcc:", normAcc.shape)

## -> Run the test
FindAnglesAndVM_TF(25, 2)
