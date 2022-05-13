"""
FP1_spSpkr.py (30%)
"""
## import modules needed below

import math
import numpy as np
import matplotlib.pyplot as plt
from lib.Audio import WAVReader, WAVWriter
from lib.DSP_Tools import findEndpoint, getPitchContour, rms, normaliseRMS
from scipy.signal import firwin, firwin2, lfilter,freqz
from numpy import arange
import os

def spSpkr(ifPlot):
    # detech the number of channels in the signal
    uttList = os.listdir("signals/utterance/")
    # Check whether gate file exist, if exist remove all
    if not os.path.exists("resources"):
        os.mkdir("resources")
    if os.path.exists("resources/gated"):
        gatedList = os.listdir("resources/gated")
        for f in gatedList:
            os.remove("resources/gated/"+f)
    else:
        os.mkdir("resources/gated/");
    for f in uttList:
        wr = WAVReader("signals/utterance/"+f)
        # convert the data type to np.array
        sigArray = wr.getData()
        sig = []
        for i in sigArray:
            sig.append(i[0])
        sig = np.array(sig)
        
        Fs = wr.getSamplingRate()
        
        # Start from here
        # Calculate the low-pass frequency
        # The low-pass frequency should be w1 = 2*55/Fs
        w1 = 2*55/Fs
        # Calculate the high-pass frequency
        # The low-pass frequency should be w2 = 2*5000/Fs
        w2 = 2*5000/Fs
        numtaps = 511
        Bs = firwin(numtaps,w1,pass_zero=False,scale=True)
        y = lfilter(Bs, 1, sig)
        t = arange(len(sig)) / Fs
        fft_bins = np.fft.fft(Bs)
        wfreqz, hfreqz = freqz(Bs)
        
        if ifPlot == 1:
            # Plot the Frequency Response
            plt.subplot(311)
            plt.plot(16000*wfreqz/(2*np.pi), 20*np.log10(np.abs(hfreqz)), label="Frequency Response")
            plt.xlim([0,512])
            plt.legend()
            
            plt.subplot(312)
            plt.plot(arange(len(sig)) / Fs,abs(np.fft.fft(sig)), label = "Original signal")
            plt.legend()
            
            plt.subplot(313)
            plt.plot(arange(len(sig)) / Fs,abs(np.fft.fft(y)), label = "Filtered signal")
            
            plt.legend()
            plt.show()
            
            # Plot the original signal and the filtered signal and isSil
            plt.subplot(211)
            plt.plot(t, sig, label = "Original signal")
            plt.legend()
            plt.subplot(212)
            plt.plot(t, y, label = "Filtered signal")
            plt.legend()
            plt.show()
        
        
        yArray = []
        for i in y:
            yArray.append([i])
        yArray = np.array(yArray)
        
        if ifPlot == 1:
            # Find End Point
            isSil = findEndpoint(sigArray, Fs, win_size=0.02, threshold_en=50, threshold_zcr=0.05)
            plt.subplot(211)
            plt.plot(isSil[0])
            
            isSil = findEndpoint(yArray, Fs, win_size=0.02, threshold_en=50, threshold_zcr=0.05)
            plt.subplot(212)
            plt.plot(isSil[0])
            plt.show()
        
        isSil = findEndpoint(yArray, Fs, win_size=0.02, threshold_en=50, threshold_zcr=0.05)
        # Cut the silence part of the signal according to isSil function
        startPoint = 0
        endPoint = len(isSil[0])-1
        for i in range(len(isSil[0])):
            if isSil[0][0] == False:
                    break
            if i >=1 :
                if isSil[0][i] == False and isSil[0][i-1] == True:
                    startPoint = i
                    break
                
        for i in range(len(isSil[0])):
            if isSil[0][len(isSil[0])-1] == False:
                break
            if len(isSil[0]) - i -2 >= 0:
                j = len(isSil[0]) - i -2
                if isSil[0][j] == False and isSil[0][j+1] == True:
                    endPoint = j
                    break
        cutSig = yArray[int(0.02*Fs*(startPoint)):int(0.02*Fs*(1+endPoint))]
        finalSig,k = normaliseRMS(cutSig,0.01)
        F0s, T = getPitchContour(finalSig, Fs, size_win=0.015, overlap=0.5, lagstep=1, win_type="rectangle", cleanup=True)
        if ifPlot == 1:
            plt.plot(arange(len(finalSig)) / Fs, finalSig, label = "Filtered signal")
            plt.legend()
        
        # Calculate the average F0s
        countF0s = 0
        F0sSum = 0
        for i in range(len(T)):
            if T[i] != 0:
                countF0s = countF0s + 1
                F0sSum = F0sSum + F0s[i]
        meanF0Sum = round(F0sSum/countF0s,2)
        print('Average F0s of '+f+' is',meanF0Sum)
        
        # Save the audio file
        if meanF0Sum < 140:
            ww = WAVWriter("resources/gated/"+f,finalSig, fs=Fs, bits=wr.getBitsPerSample())
            ww.write()