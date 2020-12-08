import numpy as np
import argparse
from Spectrogram import BaseSpectorgram
import matplotlib.pyplot as plt
def str2bool(value):
    return value == "True"

if __name__=="__main__":
    argsParser=argparse.ArgumentParser(description="Electrical signal anaylsis with STFT(Short Time Fourier Transform)")
    argsParser.add_argument("--sourcePath", default="./data/newtest.wav", help="the location of audio source")
    ##STFT related arguments
    argsParser.add_argument("--windowLen", type=float,default=20, help="length of STFT window in second")
    argsParser.add_argument("--windowShift", type=float, default=1, help="length of window shift in second")
    argsParser.add_argument("--nomFreq", type=float, default=50, help="nominal frequency of input signal")
    argsParser.add_argument("--zeroPadding", type=str2bool, default=True, help="if true, calculate fft for 2^powerOf2 points")
    argsParser.add_argument("--p2", type=int, default=16, help="power of two to be used in fft zero-padding")

    args = vars(argsParser.parse_args())
    spec=BaseSpectorgram(args)

    freqAxis,timeAxis,ss=spec.SpectogramWithScipy()
    # spec.PlotSpectogram(timeAxis,freqAxis,ss,"Scipy spec")
    fr,TM,psd,freq=spec.ManualSpectogram()
    spec.PlotSpectogram(TM, fr, psd,"My Spec")
    freq=freq

