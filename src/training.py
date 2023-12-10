import os

import librosa
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def find_closest_timestamp(timestamps: list, target: float):
    return np.argmin(np.abs(np.asarray(timestamps) - target))


# --------------------------------------------------------------------------- #
# Generate training and testing data
labels = pd.read_csv("bahnaric/labels.csv")

train, test = train_test_split(labels, test_size=0.2, random_state=42)


def gen_dataset(
    subset: pd.DataFrame, label: str, neighbor_frames: int
) -> pd.DataFrame:
    dataset = []
    for _, row in subset.iterrows():
        # Load computed features
        file_name = row["file_name"]
        features_file_name = os.path.join(
            "bahnaric/features",
            os.path.basename(row["file_name"]).replace("TextGrid", "parquet"),
        )
        if not os.path.exists(features_file_name):
            print(f"File {features_file_name} not found")
            continue
        features = pd.read_parquet(
            features_file_name.replace("TextGrid", "parquet").replace(
                "dataset", "features"
            )
        )
        # Recreate index for features
        features.index = np.arange(len(features))

        # Add file name
        features["file_name"] = file_name

        # Set label to 0
        features["label"] = 0

        # Set ground truth to 0
        features["ground_truth"] = 0

        # Translate marked labels from timestamp to frame index
        signal, sr = librosa.load(
            file_name.replace("TextGrid", "wav"), sr=16000
        )

        # Break audio into frames
        frame_length = int(sr * 0.005)  # 5ms
        hop_length = int(sr * 0.001)  # 1ms
        frames = librosa.util.frame(
            signal, frame_length=frame_length, hop_length=hop_length
        )

        # Insert frames index into features
        features["frame_index"] = np.arange(frames.shape[1])

        # Calculate the timestamp of each frame
        timestamps = np.arange(frames.shape[1]) * hop_length / sr
        assert len(timestamps) == len(features)

        # label index
        label_index = find_closest_timestamp(
            timestamps=timestamps,
            target=row["start"] if label == "ov" else row["end"],
        )
        features.loc[features.index[label_index], "ground_truth"] = 1
        # Relax the condition by allowing the label to be within
        # neighbor_frames frames
        label_index = np.arange(
            max(label_index - neighbor_frames, 0),
            min(label_index + neighbor_frames, len(timestamps)),
        )
        features.loc[features.index[label_index], "label"] = 1

        # Assuming ov and op appear in the (25%) beginning and end of the
        # timestamps, remove index in the middle
        if label == "ov":
            middle_index = np.arange(int(0.25 * len(features)), len(features))
        else:
            middle_index = np.arange(0, int(0.75 * len(features)))
        features = features.drop(features.index[middle_index])

        dataset.append(features)
    dataset = pd.concat(dataset)
    return dataset


# --------------------------------------------------------------------------- #
# Training

label = "op"
neighbor_frames = 15
training_set = gen_dataset(train, label, neighbor_frames)
testing_set = gen_dataset(test, label, neighbor_frames)

X_train = training_set.drop(
    ["label", "ground_truth", "frame_index", "file_name"], axis=1
)
y_train = training_set["label"]
print(X_train.shape, y_train.shape)

X_val = testing_set.drop(
    ["label", "ground_truth", "frame_index", "file_name"], axis=1
)
y_val = testing_set["label"]
print(X_val.shape, y_val.shape)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 30,
    "learning_rate": 0.01,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 10,
    "verbose": 0,
    "num_threads": os.cpu_count(),
}

model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_val)

# --------------------------------------------------------------------------- #
# Inference
pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
pred = pd.concat(
    [
        testing_set[
            ["label", "ground_truth", "frame_index", "file_name"]
        ].reset_index(drop=True),
        pd.DataFrame(pred_proba, columns=["proba"]),
    ],
    axis=1,
)
pred.to_csv("predictions.csv", index=False)
