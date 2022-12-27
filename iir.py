import numpy as np
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.io import wavfile
from scipy.signal import butter
import matplotlib as plt

# The Butterworth filter is a type of signal processing filter designed to have a frequency response that is as flat as possible in the passband.
