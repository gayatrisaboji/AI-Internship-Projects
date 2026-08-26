# ============================================
# WEEK 2 - TASK 01
# TEXT CLASSIFICATION
# STEP 11 - MODEL TRAINING + EVALUATION
# ============================================

import os
import pandas as pd
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================
# STEP 1: DOWNLOAD NLTK RESOURCES
# ============================================

print("\nDownloading NLTK resources...")

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# ============================================
# STEP 2: CREATE DIRECTORIES
# ============================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================
# STEP 3: LOAD DATASET
# ============================================

print("\n" + "=" * 50)
print("LOADING DATASET")
print("=" * 50)

data_path = "data/data.csv"

if not os.path.exists(data_path):
    print("\nERROR: Dataset not found!")
    print("Expected file:", data_path)
    exit()


df = pd.read_csv(data_path)

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nClass Distribution:")
print(df["label"].value_counts())


# ============================================
# STEP 4: DATA VALIDATION
# ============================================

print("\n" + "=" * 50)
print("DATA VALIDATION")
print("=" * 50)

required_columns = ["text", "label"]

for column in required_columns:

    if column not in df.columns:

        print(f"\nERROR: Missing column: {column}")
        exit()


# Remove missing values
df = df.dropna(subset=["text", "label"])

# Remove duplicate text
df = df.drop_duplicates(subset=["text"])

print("\nNumber of samples after cleaning:", len(df))


# ============================================
# STEP 5: TEXT PREPROCESSING
# ============================================

print("\n" + "=" * 50)
print("TEXT PREPROCESSING")
print("=" * 50)

stop_words = set(stopwords.words("english"))


def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove punctuation and stopwords
    tokens = [
        word
        for word in tokens
        if word.isalnum() and word not in stop_words
    ]

    # Join tokens back into a sentence
    return " ".join(tokens)


df["clean_text"] = df["text"].apply(preprocess_text)


print("\nExample preprocessing:")

for index, row in df.head(5).iterrows():

    print("\nOriginal:")
    print(row["text"])

    print("Cleaned:")
    print(row["clean_text"])


# ============================================
# STEP 6: FEATURES AND LABELS
# ============================================

X = df["clean_text"]

y = df["label"]


# ============================================
# STEP 7: TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


print("\n" + "=" * 50)
print("TRAIN / TEST SPLIT")
print("=" * 50)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# STEP 8: TF-IDF VECTORIZATION
# ============================================

print("\n" + "=" * 50)
print("TF-IDF VECTORIZATION")
print("=" * 50)

vectorizer = TfidfVectorizer(

    max_features=5000,

    ngram_range=(1, 2)
)


# Fit TF-IDF only on training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform test data using the same vectorizer
X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF completed successfully!")

print("\nTraining TF-IDF Shape:")
print(X_train_tfidf.shape)

print("\nTesting TF-IDF Shape:")
print(X_test_tfidf.shape)

print("\nNumber of Features:")
print(len(vectorizer.get_feature_names_out()))


# ============================================
# SAVE TF-IDF VECTORIZER
# ============================================

vectorizer_path = "models/tfidf_vectorizer.pkl"

joblib.dump(
    vectorizer,
    vectorizer_path
)

print("\nTF-IDF vectorizer saved successfully!")

print("Saved to:")
print(vectorizer_path)


# ============================================
# STEP 9: TRAIN MACHINE LEARNING MODEL
# ============================================

print("\n" + "=" * 50)
print("MODEL TRAINING")
print("=" * 50)

print("\nTraining Logistic Regression model...")


model = LogisticRegression(

    max_iter=1000,

    random_state=42
)


model.fit(

    X_train_tfidf,

    y_train
)


print("\nLogistic Regression training completed!")


# ============================================
# STEP 10: MODEL INFORMATION
# ============================================

print("\n" + "=" * 50)
print("MODEL INFORMATION")
print("=" * 50)

print("\nModel:")
print(model)

print("\nClasses:")
print(model.classes_)

print("\nNumber of classes:")
print(len(model.classes_))


# ============================================
# STEP 11: MAKE TEST PREDICTIONS
# ============================================

print("\n" + "=" * 50)
print("TEST PREDICTIONS")
print("=" * 50)


y_pred = model.predict(X_test_tfidf)


print("\nActual vs Predicted:")

for actual, predicted in zip(y_test, y_pred):

    print(
        f"Actual: {actual:10s} | "
        f"Predicted: {predicted}"
    )


# ============================================
# STEP 12: MODEL EVALUATION
# ============================================

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)


# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)


# Precision
precision = precision_score(
    y_test,
    y_pred,
    pos_label="positive",
    zero_division=0
)


# Recall
recall = recall_score(
    y_test,
    y_pred,
    pos_label="positive",
    zero_division=0
)


# F1 Score
f1 = f1_score(
    y_test,
    y_pred,
    pos_label="positive",
    zero_division=0
)


print("\nAccuracy:")
print(f"{accuracy:.4f} ({accuracy * 100:.2f}%)")


print("\nPrecision:")
print(f"{precision:.4f}")


print("\nRecall:")
print(f"{recall:.4f}")


print("\nF1 Score:")
print(f"{f1:.4f}")


# ============================================
# STEP 13: CLASSIFICATION REPORT
# ============================================

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)


report = classification_report(

    y_test,

    y_pred,

    zero_division=0
)


print("\n")
print(report)


# ============================================
# STEP 14: CONFUSION MATRIX
# ============================================

print("\n" + "=" * 50)
print("CONFUSION MATRIX")
print("=" * 50)


cm = confusion_matrix(

    y_test,

    y_pred,

    labels=["negative", "positive"]
)


print("\nRows = Actual")
print("Columns = Predicted")


print("\n              Predicted")
print("              Negative  Positive")


print(
    f"Actual Negative    {cm[0][0]:3d}       {cm[0][1]:3d}"
)


print(
    f"Actual Positive    {cm[1][0]:3d}       {cm[1][1]:3d}"
)


# ============================================
# STEP 15: SAVE MODEL
# ============================================

print("\n" + "=" * 50)
print("SAVING MODEL")
print("=" * 50)


model_path = "models/text_classifier_model.pkl"


joblib.dump(

    model,

    model_path
)


print("\nModel saved successfully!")

print("Saved to:")
print(model_path)


# ============================================
# STEP 16: SAVE TEST PREDICTIONS
# ============================================

test_results = pd.DataFrame({

    "text": X_test,

    "actual_label": y_test,

    "predicted_label": y_pred

})


predictions_path = "outputs/test_predictions.csv"


test_results.to_csv(

    predictions_path,

    index=False
)


print("\nTest predictions saved to:")
print(predictions_path)


# ============================================
# STEP 17: SAVE EVALUATION RESULTS
# ============================================

evaluation_results = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]

})


evaluation_path = "outputs/evaluation_results.csv"


evaluation_results.to_csv(

    evaluation_path,

    index=False
)


print("\nEvaluation results saved to:")
print(evaluation_path)


# ============================================
# COMPLETION
# ============================================

print("\n" + "=" * 50)
print("STEP 11 COMPLETED SUCCESSFULLY!")
print("=" * 50)


print("\nPipeline completed:")

print("Raw Text")
print("   ↓")
print("NLTK Preprocessing")
print("   ↓")
print("Train/Test Split")
print("   ↓")
print("TF-IDF")
print("   ↓")
print("Logistic Regression")
print("   ↓")
print("Predictions")
print("   ↓")
print("Accuracy / Precision / Recall / F1")
print("   ↓")
print("Confusion Matrix")


print("\nNext step:")
print("Visualize Model Performance")