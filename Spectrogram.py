import numpy as np
import soundfile as sf
from scipy import signal
import scipy
import matplotlib.pyplot as plt
from scipy import fft
import time
class BaseSpectorgram():

    def __init__(self,args):

            self.audioSource = args['sourcePath']
            self.windowLen = args['windowLen']
            self.shiftLen=args['windowShift']
            self.overlapLen =self.windowLen-self.shiftLen
            self.nomFreq = args['nomFreq']
            self.minSampFreq = self.nomFreq * 3
            self.isZeropadding=args['zeroPadding']
            self.isOneSided=args['oneSided']
            self.p2=args['p2']
            self.windowType=args['windowType']
            self.interpolMethod=args['interpolMethod']


            self.audioSamples = None
            self.fsOriginal = None
            self.fsNew = None

            self.freqAxis = None
            self.timeAxis = None
            self.freqResponse = None
            self.psd=None
            self.binCount=None
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
        self.window = signal.get_window(self.windowType, self.windowSize)

    def GetSamplingOrjFreq(self):
        return self.fsOriginal

    def GetAudioData(self):
        return self.audioSamples

    def ManualSpectogramWithStride(self):

        shp = ((self.decimatedSignal.shape[0]-self.overlapSize) // self.shiftSize,self.windowSize)
        strides = (self.shiftSize*self.decimatedSignal.strides[0],self.decimatedSignal.strides[0])
        sample = np.lib.stride_tricks.as_strided(self.decimatedSignal, shape=shp, strides=strides)

        sample = signal.detrend(sample, type='constant', axis=-1)
        sample *= self.window
        self.binCount = 2 ** self.p2 if self.isZeropadding else self.windowSize

        if self.isOneSided:
            result=fft.rfft(sample.real, n=self.binCount)
        else:
            result = fft(sample, self.binCount)


        scale = 1.0 / (self.fsNew * (self.window * self.window).sum())
        self.freqResponse=(result * np.sqrt(scale))

        dFs = self.fsNew / self.binCount

        if self.isOneSided:

           rng=np.arange(self.freqResponse.shape[1])
           self.freqAxis=rng*dFs
           self.psd = np.conjugate(self.freqResponse) * self.freqResponse

           if self.binCount%2:
               self.psd[...,1:]*=2
           else:
               self.psd[...,1:-1]*=2

        else:
            rng1 = np.arange(int(self.freqResponse.shape[1] / 2))
            rng2 = -np.arange(int(self.freqResponse.shape[1] / 2), 0, -1)
            self.freqAxis = np.hstack((rng1 * dFs, rng2 * dFs))
            self.psd = np.conjugate(self.freqResponse) * self.freqResponse

        self.timeAxis = np.arange(self.windowSize / 2, self.decimatedSignal.shape[0] - self.windowSize / 2, self.shiftSize) / self.fsNew

        return self.freqAxis,self.timeAxis,self.freqResponse.T,self.psd.real.T



    def ManualSpectogramWithLoop(self):

        i = 0
        freqRes=[]
        Nx = self.decimatedSignal.shape[0]
        self.binCount = 2 ** self.p2 if self.isZeropadding else self.windowSize

        while True:
            rng = range(self.shiftSize * i, self.windowSize + self.shiftSize * i)
            if self.windowSize + self.shiftSize * i >= Nx:
                break

            i += 1
            sample = self.decimatedSignal[rng]
            sample = signal.detrend(sample, type='constant', axis=-1)
            sample *= self.window

            if self.isOneSided:
                result = fft.rfft(sample.real, n=self.binCount)
            else:
                result = fft(sample, self.binCount)

            scale = 1.0 / (self.fsNew * (self.window * self.window).sum())
            freqRes.append(result*np.sqrt(scale))

        self.freqResponse=np.vstack(freqRes)
        dFs = self.fsNew / self.binCount

        if self.isOneSided:
           rng=np.arange(self.freqResponse.shape[1])
           self.freqAxis=rng*dFs
           self.psd = np.conjugate(self.freqResponse) * self.freqResponse

           if self.binCount%2:

               self.psd[...,1:]*=2
           else:
               self.psd[...,1:-1]*=2

        else:
            rng1 = np.arange(int(self.freqResponse.shape[1] / 2))
            rng2 = -np.arange(int(self.freqResponse.shape[1] / 2), 0, -1)
            self.freqAxis = np.hstack((rng1 * dFs, rng2 * dFs))
            self.psd = np.conjugate(self.freqResponse) * self.freqResponse

        self.timeAxis=np.arange(self.windowSize/2,Nx-self.windowSize/2,self.shiftSize)/self.fsNew

        return self.freqAxis,self.timeAxis,self.freqResponse.T,self.psd.real.T

    def SpectogramWithScipy(self):

        self.binCount = 2 ** self.p2 if self.isZeropadding else self.windowSize
        self.freqAxis, self.timeAxis, self.freqResponse = signal.spectrogram(self.decimatedSignal, self.fsNew
                                                                             ,nfft=self.binCount,
                                                                             window=self.window,
                                                                             noverlap=self.overlapSize, mode='complex',return_onesided=self.isOneSided)
        self.psd = np.conjugate(self.freqResponse.T) * (self.freqResponse.T)

        if self.isOneSided:
            if self.binCount % 2:
                self.psd[..., 1:] *= 2
            else:
                self.psd[..., 1:-1] *= 2

        return self.freqAxis, self.timeAxis, self.freqResponse,self.psd.real.T

    def PlotSpectogram(self,timeAxis,freqAxis,psd,plotName):
        if self.isOneSided:
            plt.pcolormesh(timeAxis, freqAxis, psd, shading='gouraud')
        else:
            plt.pcolormesh(timeAxis, fft.fftshift(freqAxis),fft.fftshift(psd, axes=0), shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(plotName)
        plt.show()

    def CalculateNominalFreq(self,inputMat):
        maxIndices=np.argmax(inputMat,axis=0)
        maxFreqs=self.freqAxis[maxIndices]
        return maxFreqs

    def PlotNominalFreq(self, freqs):
        plt.plot(freqs)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    def Interpolater(self,inputMat):

        inputMat=inputMat.T
        maxIdx=np.argmax(inputMat,axis=1)
        idx=np.arange(inputMat.shape[0])[:,np.newaxis]
        X=np.vstack((maxIdx-1,maxIdx,maxIdx+1))
        Y=inputMat[idx,X.T]
        D01=X[0]-X[1]
        D02=X[0]-X[2]
        D12=X[1]-X[2]

        if self.interpolMethod=='Polynomial':
            Y = Y.T
            A1 = (Y[1] - Y[0]) / -D01
            A2 = (-D02) * ((Y[2] - Y[1]) / -D12 - (Y[1] - Y[0]) / -D01)
            Fmax=0.5*(X[1]+X[2]-A1/A2)*self.fsNew / self.binCount

        elif self.interpolMethod=='Lagrange':
            D=np.vstack((D01*D02, -D01*D12, -D02*-D12)).T
            A=Y/D
            S=np.vstack((X[1]+X[2], X[0]+X[2], X[0]+X[1])).T
            S=-np.sum(A*S,axis=1)
            Fmax=-S/(2*np.sum(A,axis=1))*self.fsNew / self.binCount
        else:
            raise Exception("Interpolation method can be selected as Polynomial or Lagrange only")

        return Fmax






