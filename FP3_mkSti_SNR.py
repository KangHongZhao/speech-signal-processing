"""
FP3_mkSti_SNR.py (20%)
"""
## import modules needed below
from lib.Audio import WAVReader, WAVWriter
import os
import random
from lib.DSP_Tools import rms,snr
import numpy as np

## insert your code below
def setSNR(speechSig, noiseSig, targetSNR):
    ratio = np.power(rms(speechSig),2)/np.power(rms(noiseSig),2)
    factor = 1/np.power(10,targetSNR/10)
    k = np.sqrt(ratio*factor)
    noiseSig = k*noiseSig
    #print("verification SNR: ", 10*np.log10(np.power(rms(speechSig),2)/np.power(rms(noiseSig),2)))
    #print("verification SNR: ", snr(speechSig,noiseSig))
    return noiseSig, k

def mkStiSnr():
    
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
    FILE = open("etc/ssn_idx1.txt","w")

    # Check whether SNN_colocated file exist, if exist remove all
    if not os.path.exists("stimuli"):
        os.mkdir("stimuli")
    if os.path.exists("stimuli/SNN_colocated"):
        colocatedList = os.listdir("stimuli/SNN_colocated")
        for f in colocatedList:
            os.remove("stimuli/SNN_colocated/"+f)
    else:
        os.mkdir("stimuli/SNN_colocated/");

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
        noiseSigAdjust, k = setSNR(speechArray,noiseSlice,-3)
        steroWav = np.column_stack((speechArray,noiseSigAdjust))
        #print(speechArray.shape,noiseSigAdjust.shape,steroWav.shape)
            
        ww = WAVWriter("stimuli/SNN_colocated/"+f,steroWav, fs=wr.getSamplingRate(), bits=wr.getBitsPerSample())
        ww.write()
    FILE.close()
        