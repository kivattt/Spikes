import numpy as np
import scipy.signal as sps
from enum import Enum

class Algorithm(Enum):
    DEFAULT = 0
    ARGMAX = 1
    CORRELATION = 2
    SCI_PI_CORRELATION = 3
    AMP_CORRELATION = 4
    AMP_DIFF = 5
    GCCPHAT = 6

# Runs a band-reject filter on data and returns a new np.ndarray result
def band_reject(sample_freq: int, data: np.ndarray, reject_freq, bandwidth_hz):
    if sample_freq <= 0:
        raise ValueError("sample_freq must be > 0")
    if reject_freq <= 0:
        raise ValueError("reject_freq must be > 0")
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be > 0")

    nyquist = sample_freq / 2
    if reject_freq >= nyquist:
        raise ValueError("reject_freq must be below Nyquist frequency")

    q_factor = reject_freq / bandwidth_hz
    if q_factor <= 0:
        raise ValueError("Derived Q factor must be > 0")

    b, a = sps.iirnotch(reject_freq, q_factor, fs=sample_freq)

    signal = np.asarray(data, dtype=np.float64)
    if signal.size == 0:
        return signal

    min_len_for_filtfilt = 3 * max(len(a), len(b))
    if signal.shape[0] <= min_len_for_filtfilt:
        return sps.lfilter(b, a, signal, axis=0)

    return sps.filtfilt(b, a, signal, axis=0)

# Returns the generic time difference in seconds between two waveforms

def get_offset(wave1: tuple[int, np.ndarray], wave2: tuple[int, np.ndarray], algorithm: Algorithm):
    data1 = wave1[1]
    data2 = wave2[1]

    match algorithm:
        case Algorithm.ARGMAX:
            return algorithm_argmax(wave1[0], data1, wave2[0], data2)
        case Algorithm.CORRELATION:
            return algorithm_numpy_correlate(wave1[0], data1, wave2[0], data2)
        case Algorithm.SCI_PI_CORRELATION:
            return algorithm_scipy_correlate(wave1[0], data1, wave2[0], data2)
        case Algorithm.AMP_CORRELATION:
            return algorithm_amp_correlate(wave1[0], data1, wave2[0], data2)
        case Algorithm.AMP_DIFF:
            return algorithm_amp_diff(wave1[0], data1, wave2[0], data2)
        case Algorithm.GCCPHAT:
            return algorithm_gccphat(wave1[0], data1, wave2[0], data2)
        case _:
            return None

# Finds max value of both waves and calculates the time difference
def algorithm_argmax(sample_freq1: int, data1: np.ndarray, sample_freq2: int, data2: np.ndarray):
    max1InSeconds = data1.argmax() / sample_freq1
    max2InSeconds = data2.argmax() / sample_freq2

    return max1InSeconds - max2InSeconds

def algorithm_numpy_correlate(sample_freq1: int, data1: np.ndarray, sample_freq2: int, data2: np.ndarray):
    correlation = np.correlate(data1.astype(np.int64), data2.astype(np.int64), "full")
    max_index = correlation.argmax()

    return (max_index - data2.shape[0]) / sample_freq2

def algorithm_scipy_correlate(sample_freq1: int, data1: np.ndarray, sample_freq2: int, data2: np.ndarray):
    correlation = sps.correlate(data1.astype(np.int64), data2.astype(np.int64), 'full')
    max_index = correlation.argmax()

    return (max_index - data2.shape[0]) / sample_freq2

# Creates shorter volume envelopes and correlates between them
def algorithm_amp_correlate(sample_freq1: int, data1: np.ndarray, sample_freq2: int, data2: np.ndarray):
    batch_size = 200
    amplitudes1 = get_amplitude_abs_max(data1, batch_size)
    amplitudes2 = get_amplitude_abs_max(data2, batch_size)

    correlation = sps.correlate(amplitudes1.astype(np.int64), amplitudes2.astype(np.int64), 'full')
    max_index = correlation.argmax()

    return (max_index - len(amplitudes2)) * batch_size / sample_freq2

def algorithm_amp_diff(sample_freq1: int, data1: np.ndarray, sample_freq2: int, data2: np.ndarray):
    batch_size = 200
    amplitudes1 = get_amplitude_abs_max(data1, batch_size)
    amplitudes2 = get_amplitude_abs_max(data2, batch_size)

    diffs = get_diffs(amplitudes1, amplitudes2)
    min_index = diffs.argmin()

    result = min_index * batch_size / sample_freq2

    return result

# Finds the offset between data1 and data2 based on the gccphat algorithm
def algorithm_gccphat(sample_freq1: int, data1: np.ndarray, sample_freq2: int, data2: np.ndarray):
    if data1.size == 0 or data2.size == 0:
        return 0

    # Convert potential multi-channel input to mono and normalize dtype.
    ref = np.asarray(data1, dtype=np.float64)
    sig = np.asarray(data2, dtype=np.float64)
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)

    fs = sample_freq1
    if sample_freq1 != sample_freq2:
        target_len = int(round(sig.shape[0] * sample_freq1 / sample_freq2))
        if target_len <= 0:
            return 0
        sig = sps.resample(sig, target_len)

    ref_len = ref.shape[0]
    sig_len = sig.shape[0]
    fft_len = ref_len + sig_len - 1

    ref_fft = np.fft.rfft(ref, n=fft_len)
    sig_fft = np.fft.rfft(sig, n=fft_len)
    cross_spectrum = ref_fft * np.conj(sig_fft)

    magnitude = np.abs(cross_spectrum)
    magnitude[magnitude < 1e-15] = 1e-15
    correlation = np.fft.irfft(cross_spectrum / magnitude, n=fft_len)

    # Reorder to match integer lag range [-(len(sig)-1), len(ref)-1].
    correlation = np.concatenate((correlation[-(sig_len - 1):], correlation[:ref_len]))
    lags = np.arange(-(sig_len - 1), ref_len)

    lag = lags[np.argmax(np.abs(correlation))]
    return lag / fs

def get_diffs(arr1: np.ndarray, arr2: np.ndarray):
    diffs = []
    result_size = len(arr1) - len(arr2)
    if (result_size >= 0):
        r1 = arr1
        r2 = arr2
    else:
        r1 = arr2
        r2 = arr1
    for i in range(0, result_size):
        single_diff = 0
        for j in range(0, len(r2)):
            single_diff += np.int32(abs(r1[i + j] - r2[j]))
            if (single_diff > 32000):
                a = single_diff
        diffs.append(single_diff)
    return np.array(diffs)

# Other:
# - vectorized approach
# 
# average zero crossing frequency
    #sign_changes1 = np.diff(np.sign(data1))
    #zcf1 = np.count_nonzero(sign_changes1) / data1.shape[0]

# Feature methods
"""
def get_amplitude(audio: np.ndarray, window_size: int):
    amplitude = []
    for i in range(0, len(audio) - (window_size - 1)):
        audio_window = audio[i:(i + window_size)]
        max_value = max(audio_window)
        if (max_value < 0):
            max_value = - min(audio_window)
        amplitude.append(max_value)
    return amplitude
"""

def get_amplitude_max(audio: np.ndarray, batch_size: int):
    amplitude = []
    batch_count = int(len(audio) / batch_size)
    for i in range(0, batch_count):
        start = i * batch_size
        audio_window = audio[start:(start + batch_size)]
        max_value = max(audio_window)
        if (max_value < 0):
            max_value = -min(audio_window)
        amplitude.append(max_value)
    return amplitude

def get_amplitude_abs_max(audio: np.ndarray, batch_size: int):
    amplitude = []
    batch_count = int(len(audio) / batch_size)
    for i in range(0, batch_count):
        start = i * batch_size
        audio_window = audio[start:(start + batch_size)]
        max_value = np.max(np.abs(audio_window))
        amplitude.append(max_value)
        np.append(amplitude, max_value)
    amplitude2 = np.array(amplitude)
    return amplitude2
