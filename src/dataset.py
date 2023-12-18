import glob
import multiprocessing
import multiprocessing as mp
import os
import shutil
import zipfile

import gdown
import librosa
import numpy as np
import pandas as pd
import parselmouth
from praatio.utilities import textgrid_io
from tqdm.auto import tqdm
from unidecode import unidecode

from acoustic_features import extract_feature_means

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Download the Bahnaric dataset
if not os.path.exists("bahnaric.zip"):
    gdown.download(
        "https://drive.google.com/uc?id=19VZ4LLKih0DiyjIecy8ROAv3fsDw6Qs9",
        "bahnaric.zip",
    )

if os.path.exists("bahnaric"):
    shutil.rmtree("bahnaric")

# Unzip the ZIP file
with zipfile.ZipFile("bahnaric.zip", "r") as zip_file:
    zip_file.extractall()
shutil.move("Bana_1321_1650", "bahnaric/dataset/raw")
os.remove("bahnaric.zip")

# --------------------------------------------------------------------------- #
# Parse TextGrid


labels = []
for textgrid_file in glob.glob("bahnaric/dataset/*/*/*.TextGrid"):
    textgrid_data = parselmouth.Data.read(textgrid_file)
    textgrid_data.save_as_text_file("/tmp/textgrid.txt")

    try:
        # Read TextGrid and decode to UTF-8
        text = open("/tmp/textgrid.txt", "r", encoding="ISO-8859-1").read()
        text = unidecode(text, "utf-8")
        text = text.replace("\x00", "")
        textgrid_data = textgrid_io.parseTextgridStr(
            text,
            includeEmptyIntervals=False,
        )

        entries = []
        for tier in textgrid_data["tiers"]:
            for entry in tier["entries"]:
                entries.append(entry)

        # Take only single words
        if len(entries) > 0:
            if len(entries[0].label.split(" ")) == 1:
                # Extract labels
                labels.append(
                    {
                        "file_name": textgrid_file,
                        "start": entries[1].start,
                        "end": entries[1].end,
                    }
                )

    except Exception as e:
        print(e)
        print(text)
        pass

labels = pd.DataFrame(labels)
labels.to_csv("bahnaric/labels.csv", index=False)

# --------------------------------------------------------------------------- #
# Generate acoustic features


def _par_features_generator(file_name: str):
    signal, sr = librosa.load(file_name.replace("TextGrid", "wav"), sr=16000)

    feature_dfs = []
    for k in range(5):
        # Define window size
        k = 75 + (1 + k) * 10
        assert k % 2 == 1, "k must be odd"

        # Break audio into frames
        frame_length = int(sr * 0.005)  # 5ms
        hop_length = int(sr * 0.001)  # 1ms
        frames = librosa.util.frame(
            signal, frame_length=frame_length, hop_length=hop_length
        )

        # Pad frames at the beginning and end
        padding = (k - 1) // 2
        padded_frames = np.pad(frames, ((0, 0), (padding, padding)), mode="edge")

        # Calculate features on sliding window of k frames
        features = []
        for i in range(padding, len(padded_frames[0]) - padding):
            window = padded_frames[:, i - padding : i + padding + 1]
            feature = extract_feature_means(signal=window.flatten(), sr=sr)
            features.append(feature)

        features = pd.concat(features, axis=0)
        features.columns = [f"{col}_w{str(k).zfill(3)}" for col in features.columns]
        feature_dfs.append(features)

    features = pd.concat(feature_dfs, axis=1)
    features.to_parquet(
        os.path.join(
            "bahnaric/features",
            file_name.split("/")[-1].replace("TextGrid", "parquet"),
        )
    )


if not os.path.exists("bahnaric/features"):
    os.makedirs("bahnaric/features")

file_names = sorted(labels["file_name"].values.tolist())
file_names = [
    file_name
    for file_name in file_names
    if not os.path.exists(
        os.path.join(
            "bahnaric/features",
            file_name.split("/")[-1].replace("TextGrid", "parquet"),
        )
    )
]
file_names = [
    file_name
    for file_name in file_names
    if os.path.exists(file_name.replace("TextGrid", "wav"))
]

# Run _par_features_generator in parallel
num_cpus = mp.cpu_count() - 2
pool = multiprocessing.Pool(num_cpus)

# Use tqdm to show progress
with tqdm(total=len(file_names)) as pbar:

    def update_progress(*args):
        pbar.update()

    pool.map_async(_par_features_generator, file_names, callback=update_progress)
    pool.close()
    pool.join()
