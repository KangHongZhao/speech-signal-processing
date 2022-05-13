"""
FP2_mkNoise (25%)
"""
## import modules needed below
from matplotlib.backends.backend_pdf import PdfPages as pdfs
import os
import random
import numpy as np
from lib.DSP_Tools import LPC
from lib.Audio import WAVReader, WAVWriter
from scipy.signal import lfilter
from lib.DSP_Tools import normaliseRMS
import matplotlib.pyplot as plt

def save2PDF(fname, figs):
    handle = pdfs(fname)
    try:
        for fig in figs:
            handle.savefig(fig)
    except:
        print("Errors have occurred while saving PDF!")
    finally:
        handle.close()
## insert your code below

def makeSSN(DIR_sig, nb_sent = 10, duration = 30, tarRMS = 0.1):
    # default DIR_sig = "resources/gated"
    if os.path.exists(DIR_sig):
        gatedList = os.listdir(DIR_sig)
    else:
        return
    # Randomly shuffle the signals.
    random.shuffle(gatedList)
    wvList = gatedList[:nb_sent]
    
    # Calculate the average of nb_sent LPC
    sumLPC = np.zeros((101))
    for i in wvList:
        wr = WAVReader(DIR_sig+"/"+i)
        sigArray = wr.getData()
        sumLPC = sumLPC + LPC(sigArray,order=100)
    meanLPC = sumLPC/nb_sent
    
    # Generate white noise
    x = np.random.uniform(low=0.0, high=1.0, size = duration*16000)
    # Get filtered SSN
    y = lfilter([1], meanLPC, x)
    #print(len(y))
    yArray = []
    for i in y:
        yArray.append([i])
    yArray = np.array(yArray)
    finalY,k = normaliseRMS(yArray,tarRMS)

    sigY = []
    for i in finalY:
        sigY.append(i[0])
    sigY = np.array(sigY)

    # Check whether noise file exist, if exist remove all
    if not os.path.exists("noise"):
        os.mkdir("noise")
    else:
        noiseList = os.listdir("noise")
        for f in noiseList:
            os.remove("noise/"+f)
    ww = WAVWriter("noise/ssn.wav",finalY, fs=16000, bits=16)
    ww.write()
    
    # Save Figure
    finalFigures = []
    savFig = plt.figure()
    t = np.arange(len(finalY)) / 16000
    plt.subplot(211)
    plt.plot(t, sigY, label="Frequency Response")
    plt.xlim([0,10])
    plt.ylim([-0.3,0.3])
    plt.legend()
    plt.subplot(212)
    # 100 milliseconds = 0.1 second = 1600 sample points
    spec = 10*np.log10(np.abs(np.fft.fft(sigY[:1600])))
    plt.plot(np.arange(0,16000,10), spec, label="Frequency Response")
    plt.legend()

    # Check whether noise file exist, if exist remove all
    if not os.path.exists("etc"):
        os.mkdir("etc")
    else:
        noiseList = os.listdir("etc")
        for f in noiseList:
            os.remove("etc/"+f)
    finalFigures.append(savFig)
    
    # output Figure
    save2PDF("etc/T-F.pdf",finalFigures)