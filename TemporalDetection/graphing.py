import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np
import temporal_detection as td
import os

base = os.path.join("test_files", "audio")

mobile_audio = os.path.join(base, "OneClapPianoMobile.wav")
pc_audio = os.path.join(base, "OneClapStuebordPCMono")

def main():
    test4()

def get_feature(data):
    x = []
    y = []

    count = 1
    sum = 0
    is_positive = data[0] >= 0

    for d in data[1:]:
        # Crossed the 0-line >> append and start a new count.
        if (is_positive and d < 0) or (not is_positive and d >= 0):
            is_positive = not is_positive
            x.append(count)
            y.append(sum)
            count = 0
            sum = 0

        count += 1
        sum += d
    x.append(count)
    y.append(sum)

    return (x, y)

def test4():
    sample_rate1, data1 = wavfile.read(mobile_audio)
    amplitude = td.get_amplitude_abs_max(data1, 200)

    x1 = np.linspace(0, 1, len(data1))
    x2 = np.linspace(0, 1, len(amplitude))

    plt.plot(x1, data1)
    plt.plot(x2, amplitude)
    plt.show()

def test3():
    sample_rate1, data1 = wavfile.read(mobile_audio)

    amplitude = td.get_amplitude_abs_max(data1, 200)
    amplitude2 = td.get_amplitude_max(data1, 200)

    plt.plot(amplitude)
    plt.plot(amplitude2)
    plt.show()

def test2():
    data1 = [0,0,1,0,1,2,1,-1,-2,0,2,1,-1,-2,0,1]
    data2 = [0,1,2,1,-1,-2]

    data1_feature = get_feature(data1)
    data2_feature = get_feature(data2)

    # plt.plot(data1_feature[0], data1_feature[1])
    # plt.plot(data2_feature[0], data2_feature[1])
    plt.scatter(data1_feature[0], data1_feature[1])
    # plt.scatter(data2_feature[0], data2_feature[1])
    plt.show()

def test1():
    sample_rate1, data1 = wavfile.read(mobile_audio)
    sample_rate2, data2 = wavfile.read(pc_audio)

    #1.953
    start = int(1.9529 * 48000)
    n = 250
    plt.scatter(data1[:n], data2[start:start+n])
    #plt.plot(data1)

    x = []
    y = []

    for angle in range (0, 360, 1):
        x_value = np.sin(2 * np.pi * (angle / 360))
        y_value = np.cos(2 * np.pi * (angle / 360))

        x.append(x_value)
        y.append(y_value)


    #plt.plot(x, y)
    #plt.scatter(x, y)

    plt.show()

if __name__ == '__main__':
    main()