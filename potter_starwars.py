import tkinter as tk
from tkinter import Label, Button, StringVar, Tk, LEFT

import warnings
warnings.filterwarnings("ignore")

import os

import pyaudio
import wave
import librosa

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model

import time, struct, sys


app = Tk()

def process(window):
    settings = initSetting()
    frames = []
    model, estimates = loadModel()

    startRecording(window)
    frames = getAudio(window, settings, save_to_file=True)
    features = extractFeatures(window, fromfile=True, duration=settings[-1])
    input_ = normalise(window, estimates, features)

    predicted_label = predict(model, input_)
    progressText.set('Prediction: '+predicted_label)
    startRecordBtn['state'] = tk.NORMAL
    window.update()

def initSetting(chunk=1024, format=pyaudio.paInt16, channels=2, rate=44100, record_seconds=15):
    return chunk, format, channels, rate, record_seconds

def startRecording(app):
    startRecordBtn['state'] = tk.DISABLED
    app.update()
    
def getAudio(window, settings, save_to_file=True, outputfile='output.wav'):    
    CHUNK, FORMAT, CHANNELS, RATE, RECORD_SECONDS = settings
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    progressText.set('Recording ...')
    window.update() 
    
    seconds = RECORD_SECONDS + 1
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        if i == int(RATE/CHUNK * ((RECORD_SECONDS + 1 - seconds))):
            countdownText.set(seconds - 1)
            seconds -= 1
            window.update()           
        
        
        data_int = struct.unpack(str(2 * CHUNK) + 'h', data)
        volumeText.set(' '*int(np.mean(data_int)/2))
        volumeLabel.configure(bg='green', fg='white')
        time.sleep(0.01)
        window.update()       

        if seconds == 0:
            break
    stream.stop_stream()
    stream.close()
    p.terminate()

    if save_to_file:
        wf = wave.open(outputfile, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    
    countdownText.set("Time's up. You did well!")
    volumeText.set("Recording complete")
    window.update()
    return frames

def loadModel():
    modelname, estimatesname = 'psModel.h5', 'psModel.csv'
    model = load_model(modelname)
    estimates = pd.read_csv(estimatesname, sep='|')
    return model, estimates

def extractFeatures(window, fromfile=True, duration=15, frames=[]):
    progressText.set("Extracting Features ...")
    window.update()

    sr = None
    if fromfile:
        x, fs = librosa.load('output.wav', sr=sr, duration=duration)
    else:
        x, fs = librosa.load(frames, sr=sr, duration=duration)
    
    #features
    tempo, beats = librosa.beat.beat_track(x, sr=fs)
    rms = np.mean(librosa.feature.rms(x))
    chroma_stft = np.mean(librosa.feature.chroma_stft(x, fs))
    chroma_cens = np.mean(librosa.feature.chroma_cens(x, fs))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(x, fs))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(x, fs))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(x, fs))
    zcr = np.mean(librosa.feature.zero_crossing_rate(x))
    onset_detection = np.std(librosa.onset.onset_detect(x, fs))

    xi = [tempo, len(beats), np.mean(beats), 
          rms, chroma_stft, chroma_cens, spectral_centroid, 
          spectral_bandwidth, spectral_rolloff, zcr, onset_detection]
    

    return np.array(xi)

def normalise(window, estimates, features):
    progressText.set("Normalising ...")
    window.update()

    normalised_features = []
    for idx in range(features.shape[0]):
        normalised_features.append((features[idx] - estimates.iloc[idx] ['mean'])/estimates.iloc[idx]['std'])
    return np.array(normalised_features).reshape(1,11)
    

def predict(model, input_):
    prediction = model.predict(input_)[0][0]
    if prediction > 0.5:
        return "Potter"
    else:
        return "StarWars"

def createSpace(app):
    Label(app, text="").pack()




if __name__ == '__main__':
    recordWelcomeMsg = Label(app, text='Welcome', font=('bold',25))
    recordWelcomeMsg.pack()

    createSpace(app)

    recordWelcomeMsg2 = Label(app, text='Please whistle or hum for 15 seconds', font=('bold',16))
    recordWelcomeMsg2.pack()

    createSpace(app)

    #button
    startRecordBtn = Button(app, text='Start Recording', bg='blue', fg='white', font=('bold', 12), compound=LEFT, command=lambda: process(app))
    startRecordBtn.pack()

    createSpace(app)

    #labels
    countdownText = StringVar()
    countdownLabel = Label(app, textvariable=countdownText, font=('bold',16))
    countdownText.set("")
    countdownLabel.pack()

    createSpace(app)

    volumeText = StringVar()
    volumeLabel = Label(app, textvariable=volumeText, font=('bold',8))
    volumeText.set("")
    volumeLabel.pack()

    createSpace(app)

    progressText = StringVar()
    progressLabel = Label(app, textvariable=progressText, font=('bold',16))
    progressText.set("")
    progressLabel.pack()

    createSpace(app)

    app.title('Potter-Starwars Predictor by Riki Akbar (200475295) - MSc Big Data Science QMUL 2021-2022')
    app.geometry('680x350')

    app.mainloop()

