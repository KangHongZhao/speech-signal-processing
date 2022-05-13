"""
FP4_mkSti_Binaural.py (20%)
"""
## import modules needed below
from lib.Audio import WAVReader, WAVWriter
import os
import random
import numpy as np
from scipy.signal import fftconvolve
from lib.DSP_Tools import normaliseRMS
from FP3_mkSti_SNR import setSNR

## insert your code below

def mkStiBin():
    
    # Read noise ssn.wav from noise
    if os.path.exists("noise"):
        if os.path.exists("noise/ssn.wav"):
            wr = WAVReader("noise/ssn.wav")
            noiseArray = wr.getData()
        else:
            print("ssn.wav does not exist")
            return
    else:
        print("noise does not exist")
        return
    if os.path.exists("resources/gated"):
        gatedList = os.listdir("resources/gated")
    else:
        print("gated does not exist")
        return

    # Create ssn_idx1.txt
    if not os.path.exists("etc"):
        os.mkdir("etc")
    FILE = open("etc/ssn_idx2.txt","w")

    # Check whether SNN_colocated file exist, if exist remove all
    if not os.path.exists("stimuli"):
        os.mkdir("stimuli")
    if os.path.exists("stimuli/SNN_separated"):
        colocatedList = os.listdir("stimuli/SNN_separated")
        for f in colocatedList:
            os.remove("stimuli/SNN_separated/"+f)
    else:
        os.mkdir("stimuli/SNN_separated/");

    # Read wav from resources gated
    for f in gatedList:
        wr = WAVReader("resources/gated/"+f)
        speechArray = wr.getData()
        # Get wavLen, the noise length should be the same
        speechLen = len(speechArray)
        noiseLen = len(noiseArray)
        # Random start position
        startPos = random.randint(0,noiseLen-speechLen)
        # Write down the start position
        FILE.write(f+"\t"+str(startPos)+"\n")
        endPos = startPos + speechLen
        noiseSlice = noiseArray[startPos:endPos]
        
        # Read room impulse response
        wvRIRs = WAVReader("signals/IR/BRIR_s.wav")
        RIRspeechArray = wvRIRs.getData()
        wvRIRn = WAVReader("signals/IR/BRIR_n.wav")
        RIRnoiseArray = wvRIRn.getData()
        
        # Do RIR convolution
        speechConvolved = fftconvolve(speechArray,RIRspeechArray)
        noiseConvolved = fftconvolve(noiseSlice,RIRnoiseArray)
        #print(speechArray.shape,noiseSigAdjust.shape,steroWav.shape)
        speechNormalized,k1 = normaliseRMS(speechConvolved,0.01)
        noiseSigAdjust, k2 = setSNR(speechNormalized,noiseConvolved,-3)
        
        steroWav = speechNormalized + noiseSigAdjust
        ww = WAVWriter("stimuli/SNN_separated/"+f,steroWav, fs=wr.getSamplingRate(), bits=wr.getBitsPerSample())
        ww.write()
    FILE.close()
        