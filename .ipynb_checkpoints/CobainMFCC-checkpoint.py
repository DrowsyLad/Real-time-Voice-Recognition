import os
import librosa
import math
import json
import pandas as pd

# Deklarasi atribut MFCC
DATASET_PATH = "C:/Users/Lenovo/Documents/Education/Skripsi/Coding/mini_speech_commands"
JSON_PATH = "Prediksi.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 1
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

# Library untuk memproses audio
def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=2):

    # Output dari preproses dan ektraksi fitur
    data = {
        "mapping": [],
        "mfcc" : [],
        "labels": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # Iterasi untuk mengambil data yang berada pada subfolder buka dan tutup
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Memastikan subfolder
        if dirpath is not dataset_path:

            # Menyimpan subfolder didalam mapping
            semantic_label = dirpath.split("/")[-1]

            print("\nProcessing: {}".format(semantic_label))

            # Memproses seluruh audio
            for f in filenames:

                # Melakukan load file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)


                # Memproses segmen
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, d + 1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=5)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)