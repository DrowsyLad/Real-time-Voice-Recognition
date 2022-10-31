import time
import pyaudio
import math
import struct
import wave
import sys
import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import speechpy
import warnings
from tensorflow.keras.models import load_model
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
Threshold = 15

SHORT_NORMALIZE = (1.0/32768.0)
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
swidth = 2
Max_Seconds = 1
TimeoutSignal = ((RATE / chunk * Max_Seconds)+2)
silence = True
FileNameTmp = 'Out.wav'
Time=0
ac = []

def StartStream():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    return stream,p
def StopStream(stream,p):
    stream.stop_stream()
    stream.close()
    p.terminate()
def GetStream(chunk, stream):
    return stream.read(chunk)
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

def WriteSpeech(WriteData, stream, p):
    a = len(os.listdir('Temporary'))
    # StopStream(stream, p)
    FileName = 'Temporary/out_{}.wav'.format(a+1)
    # print("saving ", FileName.split('/')[1])
    wf = wave.open(FileName, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(WriteData)
    # signal, sr = librosa.load(wf)
    wf.close()
    ekstraksi_fitur()
    KeepRecording(TimeoutSignal,stream,p)


def KeepRecording(TimeoutSignal, stream, p):
    all = []

    data3 = GetStream(chunk, stream)
    rms_value = rms(data3)
    stop_speaking = False

    if (rms_value < 10):
        ac.append('1')

        print(len(ac))

    else:
        if len(ac) > 0:
            ac.remove()

    if len(ac) == 2:
        StopStream(stream, p)
        print("end record after timeout")
    lastblock = data3
    KeepRecord(TimeoutSignal, lastblock, stream,p)

def KeepRecord(TimeoutSignal, LastBlock, stream, p):
    all = []
    all.append(LastBlock)

    for i in range(0, int(TimeoutSignal)*1):
        try:
            data = GetStream(chunk, stream)
        except:
            continue
        all.append(data)
    # print("end record after timeout")
    data = b''.join(all)
    # print("write to File")
    WriteSpeech(data, stream, p)
            #I chage here (new Ident)
            # print(data)
    # silence = True
    # Time=0
    # listen(silence,Time)

def listen(silence,Time):
    stream, p = StartStream()
    print ("waiting for Speech")
    while silence:
        try:
            input = GetStream(chunk,stream)
        except:
            print("error")
            continue
        rms_value = rms(input)
        # print(rms_value)
        if (rms_value > Threshold):
            silence=False
            LastBlock=input
            # print ("hello ederwander I'm Recording....")
            KeepRecord(TimeoutSignal, LastBlock, stream, p)
        Time = Time + 1
        if (Time > TimeoutSignal*5):
            print ("Time Out No Speech Detected")
            sys.exit()

def ekstraksi_fitur():

    for f in os.listdir('Temporary'):
        signal, sr = librosa.load('Temporary/'+f)
        n_fft = 1024
        hop_length = 512
        length_data = signal.shape[0]
        MFCC = []
        signal = librosa.util.normalize(signal)
        mfcc = librosa.feature.mfcc(signal, sr, n_fft=n_fft, n_mfcc=13, hop_length=hop_length)
        # mfcc = speechpy.processing.cmvn(mfcc)
        MFCC = f''
        for e in mfcc:
            MFCC += f' {np.mean(e)}'
        scaler = StandardScaler()
        MFCCs = scaler.fit_transform(np.array(MFCC.split(), dtype='float')[:, np.newaxis]).T

        # print('berhasil ekstraksi')
        predict(MFCCs)
        delete_file(f)





def delete_file(filename):
    os.remove('Temporary/'+filename)

def predict(mfcc):
    model = load_model('Model_Akhir_fix.h5')
    tensor = tf.convert_to_tensor(mfcc)
    y_prob = model.predict(mfcc)
    # result = np.max(y_prob)
    result1 = np.argmax(y_prob)
    print(y_prob)
    # print(result1)
    if result1 == 0:
        print("Anda mengalami stress berat")
    elif result1 == 1:
        print("Anda mengalami stress ringan")
    else:
        print("Anda tidak stress")


# while stop is False:
#     if k==ord('q'):
#         stop = True

listen(silence, Time)