import numpy as np
from functools import lru_cache
from scipy.fftpack import dct

##############################################
############# UTILS ##########################
##############################################

def safe_log(x):
    """Prevents error on log(0) or log(<0)"""
    return np.log(np.clip(x, np.finfo(float).eps, None))

def split_signal(signal, window_length, window_stride, winFun):
    n_win = int(np.floor((len(signal) - window_length) / window_stride) + 1)
    frames = np.array([signal[i*window_stride: i*window_stride+window_length] for i in range(n_win)])
    if winFun == 1:
        return  frames * hamming_window(window_length)
    else:
        return frames


def delta(inp, keep_shape: bool=False):
    if keep_shape:
        return np.concatenate([np.full(inp[0].shape, 0)[np.newaxis],  inp[1:] - inp[:1]])
    else:
        return inp[1:] - inp[:1]

##############################################
############# FILTERS ########################
##############################################

@lru_cache()  # Prevents recalculating when calling with same parameters
def mel_filter(sample_rate, num_filt, input_length):
    """Returns a matrix of {num_filt} triangle filters linear on the mel scale"""
    def hertz_to_mels(f):
        return 1127. * np.log(1. + f / 700.)

    def mel_to_hertz(mel):
        return 700. * (np.exp(mel / 1127.) - 1.)    

    mels_v = np.linspace(hertz_to_mels(0), hertz_to_mels(sample_rate), num_filt + 2, True)
    hertz_v = mel_to_hertz(mels_v)
    indexes = (hertz_v * input_length / sample_rate).astype(int)

    filters = np.zeros([num_filt, input_length])

    for i, (left, middle, right) in enumerate([(indexes[x], indexes[x + 1], indexes[x + 2]) for x in range(len(indexes) - 2 )]):
        filters[i, left:middle] = np.linspace(0., 1., middle - left, False)
        filters[i, middle:right] = np.linspace(1., 0., right - middle, False)

    return filters

@lru_cache()  # Prevents recalculating when calling with same parameters
def bark_filterbanks(sample_rate, num_filt, fft_len):
    """Returns a matrix of {num_filt} triangle filters linear on the bark scale"""
    def hertz_to_bark(f):
        return 26.81 / (1 + 1960 / f) - 0.53

    def bark_to_hertz(bark):
        return 1960 * ((bark + 0.53) / (26.28 - bark))

    # Grid contains points for left center and right points of filter triangle
    # mels -> hertz -> fft indices
    grid_bark = np.linspace(hertz_to_bark(np.finfo(float).eps), hertz_to_bark(sample_rate), num_filt + 2, True)
    grid_hertz = bark_to_hertz(grid_bark)
    indexes = (grid_hertz * fft_len / sample_rate).astype(int)

    banks = np.zeros([num_filt, fft_len])

    for i, (left, middle, right) in enumerate([(indexes[x], indexes[x + 1], indexes[x + 2]) for x in range(len(indexes) - 2 )]):
        banks[i, left:middle] = np.linspace(0., 1., middle - left, False)
        banks[i, middle:right] = np.linspace(1., 0., right - middle, False)
    return banks


def preEmphasis(signal, factor=0.97):
    """ Applies preEmphasis 
    
    Keyword arguments:
    factor -- emphasis factor (default 0.97)

    """
    return signal - np.concatenate([[0], signal[:-1] * factor])

@lru_cache()
def hamming_window(sample_length):
    """ Return a hamming window of size {sample_length}"""
    return np.hamming(sample_length)

def hamming(frame):
    """ Applies hamming window on a audio frame"""
    return frame * hamming_window(len(frame))

##############################################
############# TRANSFORM ######################
##############################################

def power_spec(frames, fft_size=512) -> np.ndarray:
    """Calculates power spectrogram
    
    Keyword arguments:
    fft_size -- number of point on which to compute FFT (default 512)
    
    """
    fft = np.fft.rfft(frames, n=fft_size)
    return fft.real ** 2 + fft.imag ** 2

@lru_cache()
def power_spectrum_cutout_indexes(pow_spec_len, sample_rate, min_freq, max_freq)-> tuple:
    """ Returns array size and cutout indexes given min and max frequencies"""
    step = sample_rate / (2 * pow_spec_len)
    index_l = int(np.round(min_freq // step))
    index_h = int(np.round(max_freq // step))
    return (index_h - index_l, index_l, index_h)

def power_spec_cutout(power_spec: np.ndarray, sample_rate, min_freq, max_freq):
    """ Cut the powerspectrum outside specified frequecies """
    ps_len, index_l, index_h = power_spectrum_cutout_indexes(power_spec.shape[-1], sample_rate, min_freq, max_freq)
    return power_spec[:,index_l: index_h]

def mel_spec(frame, sample_rate, fft_size=512, num_filt=20):
    """Calculates mel spectrogram (condensed spectrogram)"""
    spec = power_spec(frame, window_stride, fft_size)
    return safe_log(np.dot(spec, filterbanks(sample_rate, num_filt, len(spec))))


##############################################
############# FEATURES #######################
##############################################

def mfe(audio, 
        sample_rate: int = 16000, 
        window_length: int = 1024, 
        window_stride: int = 512, 
        fft_size: int = 1024,
        min_freq: int = 20,
        max_freq: int = 7000,
        num_filter: int = 40,
        windowFun : int = 0, 
        preEmp : float = 0.97):
    """ Compute MEL filter energies.

        Keyword arguments:
        sample_rate -- Audio sampling rate (default 16000)
        window_length -- window size in sample (default 1024)
        window_stride -- window stride in sample (default 512)
        fft_size -- fft number of points (default 1024)
        min_freq -- minimum frequency in hertz (default 20)
        max_freq -- maximum frequency in hertz (default 7000)
        num_filter -- number of MEL bins (default 40)
        windowFun -- window function: 0- None | 1- hamming (default 0)
        preEmp -- preEmphasis factor ignored on None (default 0.97)
    """

    #Applies preEmphasis
    if preEmp is not None and preEmp > 0:
        audio = preEmphasis(audio)
    
    # Split Signal in frames
    frames = split_signal(audio, window_length, window_stride, windowFun)
    
    # Compute power spectrum
    power_spectrum = power_spec(frames, fft_size)

    # Frequency cut
    power_spectrum = power_spec_cutout(power_spectrum, sample_rate, min_freq, max_freq)

    # Create MEL bins
    filters = mel_filter(sample_rate // 2, num_filter, power_spectrum.shape[-1])

    # Convolute power spectrum with filters
    return np.dot(power_spectrum, filters.T)

def bfe(audio, 
        sample_rate, 
        window_length, 
        window_stride, 
        fft_size, 
        num_filter: int = 20, 
        windowFun : int = 0, 
        preEmp : float = 0.97):
    if preEmp is not None and preEmp > 0:
        audio = preEmphasis(audio)
    frames = split_signal(audio, window_length, window_stride, windowFun)
    power_spectrum = power_spec(frames, fft_size)
    filters = bark_filterbanks(sample_rate // 2, num_filter, fft_size // 2 + 1)
    return np.dot(power_spectrum, filters.T)

def lmfe(audio, 
         sample_rate: int = 16000, 
         window_length: int = 1024, 
         window_stride: int = 512, 
         fft_size: int = 1024,
         min_freq: int = 20,
         max_freq: int = 7000, 
         num_filter: int = 40, 
         windowFun : int = 0, 
         preEmp : float = 0.97):
    """ Compute log MEL filter Energies.

    Keyword arguments:
    sample_rate -- Audio sampling rate (default 16000)
    window_length -- window size in sample (default 1024)
    window_stride -- window stride in sample (default 512)
    fft_size -- fft number of points (default 1024)
    min_freq -- minimum frequency in hertz (default 20)
    max_freq -- maximum frequency in hertz (default 7000)
    num_filter -- number of MEL bins (default 40)
    windowFun -- window function: 0- None | 1- hamming (default 0)
    preEmp -- preEmphasis factor ignored on None (default 0.97)
    """
    # Compute MEL filter energy
    _mfe = mfe(audio, sample_rate, window_length, window_stride, fft_size, min_freq, max_freq, num_filter, windowFun, preEmp)
    
    # Log safely
    return safe_log(_mfe)

def lbfe(audio, 
         sample_rate, 
         window_length, 
         window_stride, 
         fft_size, 
         num_filter: int = 20, 
         windowFun : int = 0, 
         preEmp : float = 0.97):
    _bfe = bfe(audio, sample_rate, window_length, window_stride, fft_size, num_filter, windowFun, preEmp)
    return safe_log(_bfe)

def mfcc(audio, 
         sample_rate: int = 16000, 
         window_length: int = 1024, 
         window_stride: int = 512, 
         fft_size: int = 1024,
         min_freq: int = 20,
         max_freq: int = 7000,
         num_filter: int = 40, 
         num_coef: int = 20, 
         windowFun : int = 0, 
         preEmp : float = 0.97, 
         keep_first_value: bool = False) -> np.ndarray:
    """ Compute Mel Frequencies Cepstral Coeficient from signal. 
        
    Keyword arguments:

    sample_rate -- Audio sampling rate (default 16000)
    window_length -- window size in sample (default 1024)
    window_stride -- window stride in sample (default 512)
    fft_size -- fft number of points (default 1024)
    min_freq -- minimum frequency in hertz (default 20)
    max_freq -- maximum frequency in hertz (default 7000)
    num_filter -- number of MEL bins (default 40)
    num_coef -- number of output coeficients (default 20)
    windowFun -- window function: 0- None | 1- hamming (default 0)
    preEmp -- preEmphasis factor ignored on None (default 0.97)
    keep_first_value -- if False discard first MFCC value (default False)

    """
    # Compute log mel filter energy
    _lmfe = lmfe(np.squeeze(audio), sample_rate, window_length, window_stride, fft_size, min_freq, max_freq, num_filter, windowFun , preEmp)
    
    # Compute DCT to get MFCC
    mfccs = dct(_lmfe, norm='ortho')
    
    if keep_first_value:
        # Discard first value
        return mfccs[:,:num_coef]
    else:
        return mfccs[:,1:num_coef+1]