import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
from scipy import fft
class BaseSpectorgram():

    def __init__(self,args):

            self.audioSource = args['sourcePath'] #Audio source path
            self.windowLen = args['windowLen'] #STFT window length in seconds
            self.shiftLen=args['windowShift']  #STFT window shift length in seconds
            self.nomFreq = args['nomFreq'] #Nominal frequency of input signal
            self.minSampFreq = 3*self.nomFreq #Minimum sampling frequency
            self.isZeropadding=args['zeroPadding'] #Determines whether use zero-padding in fft calculation
            self.isOneSided=args['oneSided'] #If true,calculate one-sided frequency response [0,fs/2]
            self.p2=args['p2'] #If zero-padding activated,the number of fft samples will be selected as 2^(p2)
            self.windowType=args['windowType'] #Window type for rejecting discontinuity in singal segments
            self.interpolMethod=args['interpolMethod'] #Use specified method for quadratic interpolation (Polynomial or Lagrange)

            self.audioSamples = None #Store readed audio samples
            self.fsOriginal = None #Original sampling frequency on audio signal
            self.fsNew = None #Decimated sampling frequency

            self.freqAxis = None #Frequency values for spectrogram output
            self.timeAxis = None #Time values for spectrogram output
            self.freqResponse = None #Complex frequency response of signal
            self.psd=None #Power Spectral Density of frequency response
            self.binCount=None  #Number of points in fft calculation
            self.ReadAudio()

    def ReadAudio(self):

        #Read audio signal and take mean of channel if it has two channel
        self.audioSamples, self.fsOriginal = sf.read(self.audioSource)
        if len(self.audioSamples.shape) > 1:
            self.audioSamples = np.mean(self.audioSamples, axis=1)
        #Audio source couldn't be read
        if sum(self.audioSamples) == 0:
            raise Exception("Couldn't read audio file")

        #Determine new sampling frequency with the condition that it should be exact divisor
        #of original sampling frequency and must be greater than user specified minimum sampling frequency.
        for i in range((2 * self.nomFreq+1),self.fsOriginal):
            if self.fsOriginal%i == 0 and i>self.minSampFreq:
                self.fsNew = i
                break

        self.decRate=self.fsOriginal/self.fsNew  #Decimation rate
        self.decimatedSignal = signal.decimate(self.audioSamples, int(self.decRate),2)
        self.windowSize=int(np.floor(self.windowLen*self.fsNew)) #Window size in the number of samples for STFT calculation
        self.shiftSize=int(np.floor(self.shiftLen * self.fsNew)) #Shift size in the number of samples for STFT calculation
        self.overlapSize = self.windowSize-self.shiftSize #Overlap size in the number of samples for STFT calculation
        self.window = signal.get_window(self.windowType, self.windowSize) #Create window for rejecting discontinuity in singal segments

    #Return original sampling frequency of audio source
    def GetSamplingOrjFreq(self):
        return self.fsOriginal

    #Return original audio samples
    def GetAudioData(self):
        return self.audioSamples

    #Calculate power spectral density of input signal using numpy stride trick. This method way faster than shifting window in loop.
    #Steps:
    #1. Create a matrix with size (#TimeBins,#Frequencies) using stride trick by considering shiftsize and windowsize
    #2. Multiply this matrix with user specified window to suppress discontinuities on the singal's split points.
    #3. Calculate frequency response of all matrix (each row constitude a independent signal in fft function)
    # by depending on use specified side choice.(one-side or two-side)
    #4. Calculate power spectral density from frequency response
    #5. Determine frequency and time axises
    def ManualSpectogramWithStride(self):
        # 1. Create a matrix with size (#TimeBins,#Frequencies) using stride trick by considering shiftsize and windowsize
        shp = ((self.decimatedSignal.shape[0]-self.overlapSize) // self.shiftSize,self.windowSize)
        strides = (self.shiftSize*self.decimatedSignal.strides[0],self.decimatedSignal.strides[0])
        sample = np.lib.stride_tricks.as_strided(self.decimatedSignal, shape=shp, strides=strides)

        # 2. Multiply this matrix with user specified window to suppress discontinuities on the singal's split points.
        sample = signal.detrend(sample, type='constant', axis=-1)
        sample *= self.window
        self.binCount = 2 ** self.p2 if self.isZeropadding else self.windowSize

        # 3. Calculate frequency response of all matrix (each row constitude a independent signal in fft function)
        # by depending on use specified side choice.(one-side or two-side)
        if self.isOneSided:
            result=fft.rfft(sample.real, n=self.binCount)
        else:
            result = fft(sample, self.binCount)

        scale = 1.0 / (self.fsNew * (self.window * self.window).sum())
        self.freqResponse=(result * np.sqrt(scale))

        dFs = self.fsNew / self.binCount
        # 4. Calculate power spectral density from frequency response
        if self.isOneSided:

           rng=np.arange(self.freqResponse.shape[1])
           self.freqAxis=rng*dFs
           self.psd = np.conjugate(self.freqResponse) * self.freqResponse

           if self.binCount%2:
               self.psd[:,1:]*=2
           else:
               self.psd[:,1:-1]*=2

        else:
            rng1 = np.arange(int(self.freqResponse.shape[1] / 2))
            rng2 = -np.arange(int(self.freqResponse.shape[1] / 2), 0, -1)
            self.freqAxis = np.hstack((rng1 * dFs, rng2 * dFs))
            self.psd = np.conjugate(self.freqResponse) * self.freqResponse

        # 5. Determine frequency and time axises
        self.timeAxis = np.arange(self.windowSize / 2, self.decimatedSignal.shape[0] - self.windowSize / 2, self.shiftSize) / self.fsNew

        return self.freqAxis,self.timeAxis,self.freqResponse.T,self.psd.real.T

    # Calculate power spectral density of input signal by shifting window in a loop. This method slower because of the loop.
    # All calculation are the same with stride trick. The only difference is that  one window is considered a time and looping for every time window.
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

               self.psd[:,1:]*=2
           else:
               self.psd[:,1:-1]*=2

        else:
            rng1 = np.arange(int(self.freqResponse.shape[1] / 2))
            rng2 = -np.arange(int(self.freqResponse.shape[1] / 2), 0, -1)
            self.freqAxis = np.hstack((rng1 * dFs, rng2 * dFs))
            self.psd = np.conjugate(self.freqResponse) * self.freqResponse

        self.timeAxis=np.arange(self.windowSize/2,Nx-self.windowSize/2,self.shiftSize)/self.fsNew

        return self.freqAxis,self.timeAxis,self.freqResponse.T,self.psd.real.T

    # Calculate power spectral density of input signal using scipy's  spectogram funcion.
    def SpectogramWithScipy(self):

        self.binCount = 2 ** self.p2 if self.isZeropadding else self.windowSize
        ##We used shift size in our calculations, but scipy spectogram needs overlapSize which is equal to windowSize-shiftSize
        self.freqAxis, self.timeAxis, self.freqResponse = signal.spectrogram(self.decimatedSignal, self.fsNew
                                                                             ,nfft=self.binCount,
                                                                             window=self.window,
                                                                             noverlap=self.overlapSize, mode='complex',return_onesided=self.isOneSided)
        self.psd = np.conjugate(self.freqResponse.T) * (self.freqResponse.T)
        if self.isOneSided:
            if self.binCount % 2:
                self.psd[:, 1:] *= 2
            else:
                self.psd[:, 1:-1] *= 2

        return self.freqAxis, self.timeAxis, self.freqResponse,self.psd.real.T

    #Calculate maximum frequency components for each time point.
    def CalculateNominalFreq(self,inputMat):
        maxIndices=np.argmax(inputMat,axis=0)
        maxFreqs=self.freqAxis[maxIndices]
        return maxFreqs

    #Calculate maximum frequency components for each time point by using interpolation.
    #Instead of returning found maximum frequencies for each time point, fitt a quadratic function using the maximum point and
    #the first points to the right and left of the maximum point. The maximumum value of this quadratic function is selected as maximum
    #frequency bin and real frequency calculated using frequency resolution and frequency bin.

    def InterpolateFreqs(self,inputMat):
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
            A2 = ((Y[2] - Y[1]) / -D12 - (Y[1] - Y[0]) / -D01)/(-D02)
            Fmax=0.5*(X[0]+X[1]-A1/A2)*self.fsNew / self.binCount

        elif self.interpolMethod=='Lagrange':
            D=np.vstack((D01*D02, -D01*D12, -D02*-D12)).T
            A=Y/D
            S=np.vstack((X[1]+X[2], X[0]+X[2], X[0]+X[1])).T
            S=-np.sum(A*S,axis=1)
            Fmax=-S/(2*np.sum(A,axis=1))*self.fsNew / self.binCount
        else:
            raise Exception("Interpolation method can be selected as Polynomial or Lagrange only")

        return Fmax
    #Plot both 2d and 3d spectogram based on given psd matrix which is in shape (#TimeBins,#Frequencies)
    def PlotSpectogram(self,timeAxis,freqAxis,psd,plotName):

        if self.isOneSided:
            plt.pcolormesh(freqAxis,timeAxis, 10*np.log10(psd.T), shading='gouraud')
        else:
            plt.pcolormesh(fft.fftshift(freqAxis),timeAxis,10*np.log10(fft.fftshift(psd, axes=0).T), shading='gouraud')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Time [sec]')

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(freqAxis[:, None], timeAxis[None, :], 10*np.log10(psd),cmap=plt.get_cmap("viridis"))

    # Plot maximum frequency components for each time point.
    def PlotNominalFreq(self, freqs, plotName):
        plt.figure()
        plt.plot(freqs)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(plotName)






