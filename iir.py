import numpy as np
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.io import wavfile
from scipy.signal import butter, buttord, freqz, bilinear, lfilter
import matplotlib.pyplot as plt

'''The Butterworth filter is a type of signal processing filter designed
 to have a frequency response that is as flat as possible in the pass band.'''


if __name__ == '__main__':

    samplerate, song = wavfile.read('song.wav')
    number_of_samples = song.shape[0]
    left_channel = song[:, 0]

    # Design Butterworth filter
    # To determine the order of the filter we use the formula N = log(y/e)/log(wr/wc)
    # the cutoff frequency is 2*average_frequency
    Wc = 2 * 311 * 2 * np.pi
    Wr = samplerate / 2 * 2 * np.pi

    # butter(order of the filter, The critical frequency, The type of filter)
    N, wc = buttord(Wc, Wr, 1, 20, fs=samplerate)
    b, a = butter(N, wc, 'low', analog=True, output='ba')
    z, p = bilinear(b, a, fs=samplerate)
    w, h = freqz(z, p, fs=samplerate)

    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.grid(which='both', axis='both')
    plt.axvline(wc / (2 * np.pi), color='green')  # cutoff frequency

    filter_audio = lfilter(z, p, song[:, 0])
    filtered_FFT = abs(fft(filter_audio))

    freqs = fftfreq(number_of_samples, 1 / samplerate)
    plt.plot(freqs, filtered_FFT, 'b-', linewidth=2)
    wavfile.write("filtered_song.wav", samplerate, filter_audio.astype(np.int16))

    plt.show()


