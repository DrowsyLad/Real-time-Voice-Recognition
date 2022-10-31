import numpy as np
import pyaudio
import time
import librosa
import warnings
#warnings.filterwarnings('ignore')

num_mfcc=13
n_fft=2048
hop_length=512
FRAMES_PER_BUFFER = 3200

class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2
        self.p = None
        self.stream = None

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def MFCC(self, in_data):
        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        MFCCs = librosa.feature.mfcc(numpy_array, self.RATE, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        print(numpy_array.shape)
        return MFCCs

    def callback(self, in_data, frame_count, time_info, flag):
        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        a = self.MFCC(in_data)
        print(a)
        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active()): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
            time.sleep(2.0)


audio = AudioHandler()
audio.start()     # open the the stream
audio.mainloop()  # main operations with librosa
audio.stop()