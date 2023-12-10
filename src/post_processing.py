import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load predictions
pred = pd.read_csv("predictions.csv")

# --------------------------------------------------------------------------- #
threshold = 0.6

# Collect predictions for each file
results = []
for file_name in pred["file_name"].unique():
    chunk = pred[pred["file_name"] == file_name]

    # Find index of highest probability
    max_index = chunk["proba"].idxmax()

    # Set prediction to
    chunk["prediction"] = 0
    chunk.loc[max_index, "prediction"] = (
        1 if chunk.loc[max_index, "proba"] > threshold else 0
    )

    # Sum predictions based on label
    chunk["prediction"] = chunk.groupby("label")["prediction"].transform("sum")

    results.append(chunk[["label", "prediction"]].drop_duplicates())

# Concatenate results
results = pd.concat(results)

# Calculate metrics
labels = results["label"]
predictions = results["prediction"]

# Calculate confusion matrix components
TP = sum((labels == 1) & (predictions == 1))
TN = sum((labels == 0) & (predictions == 0))
FP = sum((labels == 0) & (predictions == 1))
FN = sum((labels == 1) & (predictions == 0))

# Calculate precision and recall
precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print("Precision: ", precision)
print("Recall: ", recall)
print("Accuracy: ", accuracy)
