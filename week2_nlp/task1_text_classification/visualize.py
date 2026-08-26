# ============================================
# WEEK 2 - TASK 01
# TEXT CLASSIFICATION
# STEP 12 - MODEL VISUALIZATION
# ============================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================

os.makedirs("outputs", exist_ok=True)


# ============================================
# LOAD TEST PREDICTIONS
# ============================================

print("\n" + "=" * 50)
print("LOADING TEST PREDICTIONS")
print("=" * 50)

predictions_path = "outputs/test_predictions.csv"

if not os.path.exists(predictions_path):

    print("\nERROR: test_predictions.csv not found!")

    print("Please run train.py first.")

    exit()


df = pd.read_csv(predictions_path)

print("\nTest predictions loaded successfully!")

print("\nNumber of test samples:")
print(len(df))


# ============================================
# EXTRACT ACTUAL AND PREDICTED LABELS
# ============================================

y_true = df["actual_label"]

y_pred = df["predicted_label"]


# ============================================
# CONFUSION MATRIX
# ============================================

print("\n" + "=" * 50)
print("CREATING CONFUSION MATRIX")
print("=" * 50)


labels = ["negative", "positive"]


cm = confusion_matrix(

    y_true,

    y_pred,

    labels=labels
)


print("\nConfusion Matrix:")

print(cm)


# ============================================
# PLOT CONFUSION MATRIX
# ============================================

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.xticks(
    range(len(labels)),
    labels
)

plt.yticks(
    range(len(labels)),
    labels
)


# Add values inside cells

for i in range(len(labels)):

    for j in range(len(labels)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14
        )


plt.colorbar()

plt.tight_layout()


confusion_path = "outputs/confusion_matrix.png"


plt.savefig(
    confusion_path,
    dpi=300
)


plt.show()


print("\nConfusion matrix saved to:")

print(confusion_path)


# ============================================
# LOAD EVALUATION RESULTS
# ============================================

print("\n" + "=" * 50)
print("LOADING EVALUATION RESULTS")
print("=" * 50)


evaluation_path = "outputs/evaluation_results.csv"


if not os.path.exists(evaluation_path):

    print("\nERROR: evaluation_results.csv not found!")

    print("Please run train.py first.")

    exit()


evaluation_df = pd.read_csv(
    evaluation_path
)


print("\nEvaluation results loaded successfully!")

print(evaluation_df)


# ============================================
# MODEL METRICS BAR CHART
# ============================================

print("\n" + "=" * 50)
print("CREATING METRICS CHART")
print("=" * 50)


metrics = evaluation_df["Metric"]

scores = evaluation_df["Score"]


plt.figure(figsize=(8, 6))


plt.bar(
    metrics,
    scores
)


plt.title(
    "Logistic Regression Performance"
)


plt.xlabel(
    "Evaluation Metric"
)


plt.ylabel(
    "Score"
)


plt.ylim(
    0,
    1
)


# Display values above bars

for i, score in enumerate(scores):

    plt.text(
        i,
        score + 0.02,
        f"{score:.2f}",
        ha="center"
    )


plt.tight_layout()


metrics_path = "outputs/model_metrics.png"


plt.savefig(
    metrics_path,
    dpi=300
)


plt.show()


print("\nModel metrics chart saved to:")

print(metrics_path)


# ============================================
# COMPLETION
# ============================================

print("\n" + "=" * 50)
print("STEP 12 COMPLETED SUCCESSFULLY!")
print("=" * 50)


print("\nGenerated files:")

print("1. outputs/confusion_matrix.png")

print("2. outputs/model_metrics.png")


print("\nNext step:")

print("Build a prediction system for new text")