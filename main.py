import numpy as np
import argparse
from Spectrogram import BaseSpectorgram
import matplotlib.pyplot as plt
import time
def str2bool(value):
    return value == "True"

if __name__=="__main__":
    argsParser=argparse.ArgumentParser(description="Electrical signal anaylsis with STFT(Short Time Fourier Transform)")
    argsParser.add_argument("--sourcePath", default="./data/newtest.wav", help="the location of audio source")
    ##STFT related arguments
    argsParser.add_argument("--windowLen", type=float,default=20, help="length of STFT window in second")
    argsParser.add_argument("--windowShift", type=float, default=19, help="length of window shift in second")
    argsParser.add_argument("--nomFreq", type=float, default=50, help="nominal frequency of input signal")
    argsParser.add_argument("--zeroPadding", type=str2bool, default=True, help="if true, calculate fft for 2^p2 points")
    argsParser.add_argument("--p2", type=int, default=18, help="power of two to be used in fft zero-padding")
    argsParser.add_argument("--oneSided", type=str2bool, default=True, help="If true, calculate one-sided fft")
    argsParser.add_argument("--windowType", default='hamming', help="window type for stft calculation")
    argsParser.add_argument("--interpolMethod", default='Lagrange', help="Interpolation method can be set to Polynomial or Lagrange ")
    args = vars(argsParser.parse_args())
    spec=BaseSpectorgram(args)

    # c1=time.time()
    # freqAxis1,timeAxis1,freqRes1,psd1=spec.ManualSpectogramWithStride()
    # print("Time for stride",time.time()-c1)
    #
    # c2=time.time()
    # freqAxis2,timeAxis2,freqRes2,psd2=spec.SpectogramWithScipy()
    # print("Time for scipy",time.time()-c2)
    # spec.PlotSpectogram(timeAxis2, freqAxis2, psd2,"My Spec")
    c3=time.time()
    freqAxis3,timeAxis3,freqRes3,psd3=spec.ManualSpectogramWithLoop()
    print("Time for loop",time.time()-c3)
    freqs=spec.Interpolater(psd3)
    freqs1=spec.CalculateNominalFreq(psd3)
    # freqs2 = spec.CalculateNominalFreq(psd2)
    # freqs33 = spec.CalculateNominalFreq(psd3)
    spec.PlotNominalFreq(freqs1)
    # spec.PlotNominalFreq(freqs)
    # spec.PlotSpectogram(TM, fr, psd2,"My Spec")


