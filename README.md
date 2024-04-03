# Audio Classification Project

## Overview
This project focuses on classifying audio samples into two categories: "OK" and "Not OK". The audio samples represent different parts produced by a company, and accurate classification is crucial for quality control purposes.

## Key Concepts

### Signal Processing Techniques
- **Fourier Transform**: Converts a signal from its time domain representation to the frequency domain, allowing us to analyze its frequency components.
- **Mel Frequency Cepstrum Coefficients (MFCCs)**: Representations of the short-term power spectrum of a sound, which are derived from the Mel scale of human auditory perception.
- **Filter Bank Coefficients**: Captures the frequency content of audio signals by dividing the frequency spectrum into multiple filter banks.
- **Envelopes**: Extracts the envelope of a signal, representing its slowly varying amplitude.

### Machine Learning Models
- **Convolutional Neural Network (CNN)**: A deep learning architecture particularly effective for analyzing grid-like data, such as images or time-series data, by using convolutional layers to automatically learn spatial hierarchies of features.
- **Recurrent Neural Network (RNN)**: A type of neural network architecture designed to work with sequence data, allowing information to persist over time through recurrent connections.

## Libraries Used
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `librosa`
- `keras`
- `python_speech_features`
- `tqdm`

## Project Structure
- `audio.csv`
- `clean/`
- `models/`
- `pickles/`
- `wavfiles/`
- `cfg.py`
- `preprocessing.py`
- `model.py`
- `predict.py`
- `README.md`


## Usage
1. **Data Preprocessing**:
   - Ensure that audio samples are stored in the `wavfiles/` directory.
   - Run `preprocessing.py` to preprocess the audio samples and extract features.
2. **Model Training**:
   - Run `model.py` to train machine learning models on the extracted features.
3. **Prediction**:
   - Once the models are trained, run `predict.py` to make predictions on new audio samples.

## Authors
- [Shashwat Agarwal](https://www.linkedin.com/in/shashwattagrawal/)
- [Karandeep Saluja](https://www.linkedin.com/in/karandeep-saluja-45949b221/)


