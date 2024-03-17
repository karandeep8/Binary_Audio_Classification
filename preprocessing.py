import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa


def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=True, figsize=(20, 5))

    fig.suptitle('Time Series', size=16)

    i = 0
    for ax in axes:
        ax.set_title(list(signals.keys())[i])
        ax.plot(list(signals.values())[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        i += 1


def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=True, figsize=(20, 5))

    fig.suptitle('Fourier Transform', size=16)

    i = 0
    for ax in axes:
        data = list(fft.values())[i]
        Y, freq = data[0], data[1]
        ax.set_title(list(fft.keys())[i])
        ax.plot(freq, Y)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        i += 1


def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=True, figsize=(20, 5))

    fig.suptitle('Filter Bank Coefficients', size=16)

    i = 0
    for ax in axes:
        ax.set_title(list(fbank.keys())[i])
        ax.imshow(list(fbank.values())[i], cmap='hot', interpolation='nearest')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        i += 1


def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)

    i = 0
    for ax in axes:
        ax.set_title(list(mfccs.keys())[i])
        ax.imshow(list(mfccs.values())[i], cmap='hot', interpolation='nearest')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        i += 1


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1 / rate)
    Y = abs(np.fft.rfft(y) / n)
    return (Y, freq)


# Reading the csv files
df = pd.read_csv('audio.csv')

# Getting to know the distribution of each class data
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('wavfiles/' + f)

    # Positional Based Indexing
    # Allowing us to access individual elements
    # Giving us the length of audio in seconds
    df.at[f, 'length'] = signal.shape[0] / rate

# Finding out the unique labels present in the dataset
classes = list(np.unique(df.label))
# Grouping according to labels then accessing only their length and calculating their mean
class_dist = df.groupby(['label'])['length'].mean()

# Plotting the above calculated Mean
fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
# autopct is just to set the floating values to nearby decimal place
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
plt.show()
df.reset_index(inplace=True)

signals = {}
fft = {}
fbank = {}
mfccs = {}

for c in classes:
    # dataframe will contain all the values that is 20 in our case, and then we only want first position file
    wav_file = df[df.label == c].iloc[0, 0]
    signal, rate = librosa.load('wavfiles/' + wav_file, sr=44100)
    mask = envelope(signal, rate, 0.0005)
    signal = signal[mask]
    signals[c] = signal
    fft[c] = calc_fft(signal, rate)

    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
    fbank[c] = bank
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
    mfccs[c] = mel

plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()

plot_fbank(fbank)
plt.show()

plot_mfccs(mfccs)
plt.show()

if len(os.listdir('clean')) == 0:
    for f in tqdm(df.fname):
        signal, rate = librosa.load('wavfiles/'+f, sr=16000)
        mask = envelope(signal, rate, 0.0005)
        wavfile.write(filename='clean/'+f, rate=rate, data=signal[mask])



