import numpy as np
import soundfile as sf
from scipy import signal
import scipy
from scipy.fft import fft
import matplotlib.pyplot as plt

class BaseSpectorgram():

    def __init__(self,args):

            self.audioSource = args['sourcePath']
            self.windowLen = args['windowLen']
            self.shiftLen=args['windowShift']
            self.overlapLen =self.windowLen-self.shiftLen
            self.nomFreq = args['nomFreq']
            self.minSampFreq = self.nomFreq * 3

            self.audioSamples = None
            self.fsOriginal = None
            self.fsNew = None

            self.freqAxis = None
            self.timeAxis = None
            self.freqResponse = None
            self.psd=None
            self.ReadAudio()

    def ReadAudio(self):

        self.audioSamples, self.fsOriginal = sf.read(self.audioSource)
        if len(self.audioSamples.shape) > 1:
            self.audioSamples = np.mean(self.audioSamples, axis=1)

        if sum(self.audioSamples) == 0:
            raise Exception("Couldn't read audio file")

        for i in range((2 * self.nomFreq+1),self.fsOriginal):
            if self.fsOriginal%i == 0 and i>self.minSampFreq:
                self.fsNew = i
                break

        self.decRate=self.fsOriginal/self.fsNew
        self.decimatedSignal = signal.decimate(self.audioSamples, int(self.decRate),5)
        self.windowSize=int(np.floor(self.windowLen*self.fsNew))
        self.shiftSize=int(np.floor(self.shiftLen * self.fsNew))
        self.overlapSize = self.windowSize-self.shiftSize

    def GetSamplingOrjFreq(self):
        return self.fsOriginal

    def GetAudioData(self):
        return self.audioSamples

    def ManualSpectogram(self):
        i = 0
        spec = []
        freqRes=[]
        Nx = self.decimatedSignal.shape[0]

        while True:

            rng = range(self.shiftSize * i, self.windowSize + self.shiftSize * i)
            if self.windowSize + self.shiftSize * i >= Nx:
                break
            i += 1
            sample = self.decimatedSignal[rng]
            sample = signal.detrend(sample, type='constant', axis=-1)
            sample *= np.hamming(sample.shape[0])
            sample = sample.real
            result = scipy.fft.rfft(sample, n=self.windowSize)
            freqRes.append(result)

            ##Calculate power spectrum density(psd)
            win = signal.get_window('hamming', self.windowSize)
            scale = 1.0 / (self.fsNew * (win * win).sum())
            result = np.conjugate(result) * result
            result *= scale

            if self.windowSize % 2:
                result[..., 1:] *= 2
            else:
                # Last point is unpaired Nyquist freq point, don't double
                result[..., 1:-1] *= 2

            spec.append(result.real)

            # step = nperseg - noverlap
            # shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
            # strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
            # result = np.lib.stride_tricks.as_strided(x, shape=shape,
            #                                          strides=strides)

        self.psd = np.vstack(spec).T
        self.freqResponse=np.vstack(freqRes).T

        dFs=self.fsNew/self.windowSize
        rng=np.arange(self.freqResponse.shape[0])
        self.freqAxis=rng*dFs
        self.timeAxis=np.arange(self.windowSize/2,Nx-self.windowSize/2,self.shiftSize)/self.fsNew

        return self.freqAxis,self.timeAxis,self.psd,self.freqResponse

    def SpectogramWithScipy(self):
        self.freqAxis, self.timeAxis, self.psd = signal.spectrogram(self.decimatedSignal, self.fsNew,
                                                                             window=signal.get_window('hamming',
                                                                                                      self.windowSize),
                                                                             noverlap=self.overlapSize, mode='psd')
        return self.freqAxis, self.timeAxis, self.psd

    def PlotSpectogram(self,timeAxis,freqAxis,psd,plotName):
        plt.pcolormesh(timeAxis, freqAxis, psd, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(plotName)
        plt.show()

    # def PlotNominalFreq(self):


