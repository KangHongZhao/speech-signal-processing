"""
FP5_RUN.py (5%)
"""
## import modules needed below
import FP1_spSpkr
import FP2_mkNoise
import FP3_mkSti_SNR
import FP4_mkSti_Binaural

## insert your code below
if __name__ == '__main__':
# FP1_spSpkr.py
# Start processing speaker speration, 0 for not plot
    print("FP1 running...")
    FP1_spSpkr.spSpkr(0)
# FP2_mkNoise.py
    print("FP2 running...")
    FP2_mkNoise.makeSSN("resources/gated", 50, 60, 0.05)
# FP3_mkSti_SNR.py
    print("FP3 running...")
    FP3_mkSti_SNR.mkStiSnr()
# FP4_mkSti_Binaural.py
    print("FP4 running...")
    FP4_mkSti_Binaural.mkStiBin()