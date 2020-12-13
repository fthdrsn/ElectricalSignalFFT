
import argparse
from Spectrogram import BaseSpectorgram
import matplotlib.pyplot as plt
import time

def str2bool(value):
    return value == "True"

if __name__=="__main__":

    argsParser=argparse.ArgumentParser(description="Electrical signal anaylsis with STFT(Short Time Fourier Transform)")
    argsParser.add_argument("--sourcePath", default="./data/newtest.wav", help="the location of audio source")
    argsParser.add_argument("--windowLen", type=float,default=20, help="length of STFT window in second")
    argsParser.add_argument("--windowShift", type=float, default=1, help="length of window shift in second")
    argsParser.add_argument("--nomFreq", type=float, default=50, help="nominal frequency of input signal")
    argsParser.add_argument("--zeroPadding", type=str2bool, default=True, help="if true, calculate fft for 2^p2 points")
    argsParser.add_argument("--p2", type=int, default=18, help="power of two to be used in fft zero-padding")
    argsParser.add_argument("--oneSided", type=str2bool, default=True, help="If true, calculate one-sided fft")
    argsParser.add_argument("--windowType", default='hamming', help="window type for stft calculation")
    argsParser.add_argument("--useInterpolation", type=str2bool, default=False, help="If true, use interpolation for nominal frequency calculation")
    argsParser.add_argument("--interpolMethod", default='Polynomial', help="Interpolation method can be set to Polynomial or Lagrange ")
    argsParser.add_argument("--spectogramMethod", default='ManualStride',help="Determine spectogram calculation method (ManualLoop, ManualStride, ScipySpec) ")
    argsParser.add_argument("--drawSpectogram",type=str2bool, default='False',help="If true, draw spectrogram")
    argsParser.add_argument("--drawNomFreq", type=str2bool, default='True', help="If true, draw calculated nominal frequencies")

    args = vars(argsParser.parse_args())
    spec=BaseSpectorgram(args)


    tic=time.time()
    if args['spectogramMethod']=='ManualLoop':
        freqAxis, timeAxis, freqResponse, psd = spec.ManualSpectogramWithLoop()
    elif args['spectogramMethod']=='ManualStride':
        freqAxis, timeAxis, freqResponse, psd = spec.ManualSpectogramWithStride()
    elif args['spectogramMethod']=='ScipySpec':
        freqAxis, timeAxis, freqResponse, psd = spec.SpectogramWithScipy()
    else:
        raise Exception("Please choose one of valid methods (ManualLoop,ManualStride,ScipySpec)")
    toc=time.time()-tic

    print(f"Passed time for {args['spectogramMethod']} method is {toc} second")

    if args['useInterpolation']:
        freqs = spec.InterpolateFreqs(psd)
    else:
        freqs = spec.CalculateNominalFreq(psd)

    if args['drawSpectogram']:
        spec.PlotSpectogram(timeAxis, freqAxis, psd, f"Spectogram using {args['spectogramMethod']} method")
    if args['drawNomFreq']:
        spec.PlotNominalFreq(freqs,f"Calculated frequencies using {args['spectogramMethod']} method")

    plt.show()

