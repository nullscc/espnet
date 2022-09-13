import librosa
import numpy as np
import scipy.signal
import torch

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

def wav2stftspec(y, n_fft=320, hop_length=160, window_type="hamming", win_length=None, normalize=True):
    if not win_length:
        win_length = n_fft
    # STFT
    window = windows.get(window_type, " ")
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)
    # S = log(S+1)
    spect = np.log1p(spect)
    # spect = torch.FloatTensor(spect)

    if normalize:
        mean = spect.mean()
        std = spect.std()
        spect -= mean
        spect = np.true_divide(spect, std)

    return spect

