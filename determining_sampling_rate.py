from scipy.io import wavfile

# Read the WAV file
sampling_rate, data = wavfile.read('1_not_ok.wav')

print("Sampling rate:", sampling_rate, "Hz")
