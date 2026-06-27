import pytest
import scipy.io.wavfile as wavfile
import temporal_detection as td
import time
import os

base = os.path.join("test_files", "audio")

mobile_audio = os.path.join(base, "OneClapPianoMobile.wav")
pc_audio = os.path.join(base, "OneClapStuebordPCMono.wav")

@pytest.mark.parametrize("wavfile1, wavfile2, full_expected_offset, algorithm", [
        # < 1 second
        (mobile_audio, pc_audio, 1.953, td.Algorithm.ARGMAX),
        # 7 minutes
        (mobile_audio, pc_audio, 1.953, td.Algorithm.CORRELATION),
        # < 1 second
        (mobile_audio, pc_audio, 1.953, td.Algorithm.SCI_PI_CORRELATION),
        # ? seconds
        (mobile_audio, pc_audio, 1.953, td.Algorithm.AMP_CORRELATION),
        # ? seconds
        (mobile_audio, pc_audio, 1.953, td.Algorithm.GCCPHAT)
    ])
def test_get_offset(wavfile1, wavfile2, full_expected_offset, algorithm: td.Algorithm):
    wave1 = wavfile.read(wavfile1)

    sample_rate2, wave2 = wavfile.read(wavfile2)
    wave2_length = wave2.shape[0]
    err = check_window(wave1, wave2, 0, wave2_length, sample_rate2, full_expected_offset, algorithm)
    assert err == None

@pytest.mark.parametrize("wavfile1, wavfile2, full_expected_offset, algorithm", [
        #('test_files\\audio\\OneClapPianoMobile.wav', 'test_files\\audio\\OneClapStuebordPCMono.wav', 1.953, td.Algorithm.CORRELATION),
        # 4 seconds, error = 47%
        (mobile_audio, pc_audio, 1.953, td.Algorithm.SCI_PI_CORRELATION),
        # 24 seconds, error = 60%
        (mobile_audio, pc_audio, 1.953, td.Algorithm.AMP_CORRELATION),
        # 3 minutes, error = 63%
        (mobile_audio, pc_audio, 1.953, td.Algorithm.AMP_DIFF),
        # 28 seconds, error = 60%
        (mobile_audio, pc_audio, 1.953, td.Algorithm.GCCPHAT)
    ])
def test_get_offset_accuracy_and_speed(wavfile1, wavfile2, full_expected_offset, algorithm: td.Algorithm):
    errors = []
    count_total = 0

    wave1 = wavfile.read(wavfile1)
    # sample_rate2, wave2 = wavfile.read(wavfile2)
    # wave2_length = wave2.shape[0]

    s_rate2, w2 = wavfile.read(wavfile2)
    w_length = w2.shape[0]

    sample_rate2 = s_rate2
    wave2 = td.band_reject(sample_rate2, w2, 5, 5)
    wave2_length = wave2.shape[0]

    start_time = time.perf_counter()

    # Cut from the start of wave2
    for i in range(0, 10):
        start = int(wave2_length * i / 10)
        end = wave2_length
        expected_offset = full_expected_offset + start / sample_rate2

        err = check_window(wave1, wave2, start, end, sample_rate2, expected_offset, algorithm)

        if err != None:
            errors.append(err)
        count_total += 1

    # Cut from the end of wave2
    for i in range(10, 0, -1):
        start = 0
        end = int(wave2_length * i / 10)
        expected_offset = full_expected_offset

        err = check_window(wave1, wave2, start, end, sample_rate2, expected_offset, algorithm)

        if err != None:
            errors.append(err)
        count_total += 1

    # Pan wave2 with a 1/10th window
    for i in range(0,10):
        start = int(wave2_length * i / 10)
        end = int(start + wave2_length / 10)
        expected_offset = full_expected_offset + start / sample_rate2

        err = check_window(wave1, wave2, start, end, sample_rate2, expected_offset, algorithm)

        if err != None:
            errors.append(err)
        count_total += 1

    end_time = time.perf_counter()

    execution_time = end_time - start_time

    err_count = len(errors)
    err_percentage = err_count / count_total

    assert err_percentage < 0.1, f"Errors: {err_percentage:,.2%} ({err_count}/{count_total}). Execution time: {execution_time:.4f} seconds"

def check_window(wave1, wave2, start, end, sample_rate2, expected_offset, algorithm = td.Algorithm.ARGMAX):
    wave3 = wave2[start:end]

    offset_in_seconds = td.get_offset(wave1, (sample_rate2, wave3), algorithm)

    if offset_in_seconds != pytest.approx(expected_offset, rel=0.01):
        start_time = start / sample_rate2
        end_time = end / sample_rate2
        return f"Wavfile2 window ({start_time:,.3f},{end_time:,.3f}). Expected offset is {expected_offset:,.3f}, but actual was {offset_in_seconds:,.3f}"

    return None

if __name__ == '__main__':
    pytest.main
