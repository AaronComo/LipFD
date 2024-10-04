import os
import cv2
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from librosa import feature as audio


"""
Structure of the AVLips dataset:
AVLips
├── 0_real
├── 1_fake
└── wav
    ├── 0_real
    └── 1_fake
"""

############ Custom parameter ##############
N_EXTRACT = 10   # number of extracted images from video
WINDOW_LEN = 5   # frames of each window
MAX_SAMPLE = 100 

audio_root = "./AVLips/wav"
video_root = "./AVLips"
output_root = "./datasets/AVLips"
############################################

labels = [(0, "0_real"), (1, "1_fake")]

def get_spectrogram(audio_file):
    data, sr = librosa.load(audio_file)
    mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)
    plt.imsave("./temp/mel.png", mel)


def run():
    i = 0
    for label, dataset_name in labels:
        if not os.path.exists(dataset_name):
            os.makedirs(f"{output_root}/{dataset_name}", exist_ok=True)

        if i == MAX_SAMPLE:
            break
        root = f"{video_root}/{dataset_name}"
        video_list = os.listdir(root)
        print(f"Handling {dataset_name}...")
        for j in tqdm(range(len(video_list))):
            v = video_list[j]
            # load video
            video_capture = cv2.VideoCapture(f"{root}/{v}")
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            # select 10 starting point from frames
            frame_idx = np.linspace(
                0,
                frame_count - WINDOW_LEN - 1,
                N_EXTRACT,
                endpoint=True,
                dtype=np.uint8,
            ).tolist()
            frame_idx.sort()
            # selected frames
            frame_sequence = [
                i for num in frame_idx for i in range(num, num + WINDOW_LEN)
            ]
            frame_list = []
            current_frame = 0
            while current_frame <= frame_sequence[-1]:
                ret, frame = video_capture.read()
                if not ret:
                    print(f"Error in reading frame {v}: {current_frame}")
                    break
                if current_frame in frame_sequence:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    frame_list.append(cv2.resize(frame, (500, 500)))  # to floating num
                current_frame += 1
            video_capture.release()

            # load audio
            name = v.split(".")[0]
            a = f"{audio_root}/{dataset_name}/{name}.wav"

            group = 0
            get_spectrogram(a)
            mel = plt.imread("./temp/mel.png") * 255  # load spectrogram (int)
            mel = mel.astype(np.uint8)
            mapping = mel.shape[1] / frame_count
            for i in range(len(frame_list)):
                idx = i % WINDOW_LEN
                if idx == 0:
                    try:
                        begin = np.round(frame_sequence[i] * mapping)
                        end = np.round((frame_sequence[i] + WINDOW_LEN) * mapping)
                        sub_mel = cv2.resize(
                            (mel[:, int(begin) : int(end)]), (500 * WINDOW_LEN, 500)
                        )
                        x = np.concatenate(frame_list[i : i + WINDOW_LEN], axis=1)
                        # print(x.shape)
                        # print(sub_mel.shape)
                        x = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)
                        # print(x.shape)
                        plt.imsave(
                            f"{output_root}/{dataset_name}/{name}_{group}.png", x
                        )
                        group = group + 1
                    except ValueError:
                        print(f"ValueError: {name}")
                        continue
            # print(frame_sequence)
            # print(frame_count)
            # print(mel.shape[1])
            # print(mapping)
            # exit(0)
        i += 1


if __name__ == "__main__":
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)
    if not os.path.exists("./temp"):
        os.makedirs("./temp", exist_ok=True)
    run()
