from dis import disassemble
from logging.config import listen
from xml.etree.ElementPath import prepare_predicate
import librosa
import numpy as np
import audioop
import pyaudio
import wave
import speech_recognition as sr
import os
import math
import warnings
import struct
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
FRAMES = []
SHORT_NORMALIZE = (1.0/32768.0)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 2
swidth = 2
CHUNK = 1024
SAMPLES_TO_CONSIDER = 22050
TEMPORARY_WAVE_FILENAME = "temp.wav"
SAVED_MODEL_PATH = "Coding/bahasa inggris.h5"
silence = True
Threshold = 20

mic = sr.Microphone(device_index=0)
rec = sr.Recognizer()
audio = pyaudio.PyAudio()

class _Keyword_Spotting_Service:

    model = None
    _mapping = [
        "Go",
        "Left",
        "Right",
        "Stop"
    ]
    _instance = None


    def predict(self, file_path):

        # extract MFCC
        MFCCs = self.preprocess(file_path)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        print(predictions)
        predicted_keyword = self._mapping[predicted_index]  
        return predicted_keyword


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):

        # load audio file
        signal, sample_rate = librosa.load(file_path)
        length = librosa.get_duration(signal)
        print('signal before padding', signal.shape)
        if length != 2:
            signal = librosa.util.fix_length(signal, 22050)
            print("Signal after padding", signal.shape)
        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        return MFCCs.T

def Keyword_Spotting_Service():

    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance

def rms(frame):
    count = len(frame)/swidth
    format = "%dh"%(count)
    # short is 16 bit int
    shorts = struct.unpack( format, frame )

    sum_squares = 0.0
    for sample in shorts:
        n = sample * SHORT_NORMALIZE
        sum_squares += n*n
    # compute the rms
    rms = math.pow(sum_squares/count,0.5)
    return rms * 1000

def recording(lastblock, stream):
    global FRAMES, FORMAT, CHUNK, CHANNELS, RATE, TEMPORARY_WAVE_FILENAME
    try:
        arr = []
        arr.append(lastblock)
        print ("recording...")
        # stream = audio.open(format=FORMAT, channels=CHANNELS,
        #                     rate=RATE, input=True,
        #                     frames_per_buffer=CHUNK)
        #for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        while True:
            data = stream.read(CHUNK)
            mx = audioop.max(data, 2)
            # print mx
            rms_value = rms(data)
            if rms_value < Threshold:
                break
            else:
                arr.append(data)

        print ("Finish recordings")
        # print(len(arr))

        stream.stop_stream()
        stream.close()
        # audio.terminate()
        if os.path.exists(TEMPORARY_WAVE_FILENAME):
            print("File deleted")
            # os.remove(TEMPORARY_WAVE_FILENAME)
        waveFile = wave.open(TEMPORARY_WAVE_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(arr))
        waveFile.close()
        del stream
    except Exception as e:
        print (e)
        raise

def recognise():
    # global FRAMES
    kss = Keyword_Spotting_Service()
    r = sr.Recognizer()
    with sr.AudioFile(TEMPORARY_WAVE_FILENAME) as source:
        audio = r.record(source)  # read the entire audio file
    keyword = kss.predict(TEMPORARY_WAVE_FILENAME)
    # os.remove(TEMPORARY_WAVE_FILENAME)
    try: 
        print("Perintah: " + keyword)
    except sr.UnknownValueError:  # speech is unintelligible
        print("Tidak dapat menerjemahkan")

def listen(silence):
    #stream, p = StartStream()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
    print ("waiting for Speech")
    while silence:
        try:
            input = stream.read(CHUNK)
        except:
            print("error")
            continue
        rms_value = rms(input)
        # print(rms_value)
        if (rms_value > Threshold):
            silence=False
            LastBlock=input
            # print ("hello ederwander I'm Recording....")
            recording(LastBlock, stream)
        
while True:
    listen(silence)
    recognise()
    
