import soundfile as sf
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from Spectrogram import  BaseSpectorgram
from scipy.fft import fft
from numpy.lib.stride_tricks import as_strided

seq = np.random.normal(size=10000) + np.arange(10000)
window=10
stride = seq.strides[0]
sequence_strides = as_strided(seq, shape=[len(seq) - window + 1, window], strides=[stride, stride])
sequence_strides.mean(axis=1)