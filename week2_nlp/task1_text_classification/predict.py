# ============================================
# WEEK 2 - TASK 01
# TEXT CLASSIFICATION
# STEP 13 - PREDICTION SYSTEM
# ============================================

import os
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ============================================
# STEP 1: DOWNLOAD NLTK RESOURCES
# ============================================

print("\nLoading NLTK resources...")

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# ============================================
# STEP 2: MODEL PATHS
# ============================================

model_path = "models/text_classifier_model.pkl"

vectorizer_path = "models/tfidf_vectorizer.pkl"


# ============================================
# STEP 3: CHECK MODEL FILES
# ============================================

print("\n" + "=" * 50)
print("CHECKING SAVED MODELS")
print("=" * 50)


if not os.path.exists(model_path):

    print("\nERROR: Classification model not found!")

    print("Expected:")
    print(model_path)

    print("\nPlease run:")
    print("python train.py")

    exit()


if not os.path.exists(vectorizer_path):

    print("\nERROR: TF-IDF vectorizer not found!")

    print("Expected:")
    print(vectorizer_path)

    print("\nPlease run:")
    print("python train.py")

    exit()


print("\nClassification model found!")
print("TF-IDF vectorizer found!")


# ============================================
# STEP 4: LOAD MODEL
# ============================================

print("\nLoading trained model...")

model = joblib.load(model_path)

vectorizer = joblib.load(vectorizer_path)


print("Model loaded successfully!")
print("Vectorizer loaded successfully!")


# ============================================
# STEP 5: LOAD STOPWORDS
# ============================================

stop_words = set(
    stopwords.words("english")
)


# ============================================
# STEP 6: TEXT PREPROCESSING FUNCTION
# ============================================

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove punctuation and stopwords
    tokens = [

        word

        for word in tokens

        if word.isalnum()
        and word not in stop_words

    ]

    # Convert back to string
    return " ".join(tokens)


# ============================================
# STEP 7: PREDICTION FUNCTION
# ============================================

def predict_sentiment(text):

    # Preprocess text
    clean_text = preprocess_text(text)

    # Convert text into TF-IDF features
    text_tfidf = vectorizer.transform(
        [clean_text]
    )

    # Predict class
    prediction = model.predict(
        text_tfidf
    )[0]

    # Get probabilities
    probabilities = model.predict_proba(
        text_tfidf
    )[0]

    # Get class names
    classes = model.classes_

    # Create probability dictionary
    probability_dict = dict(
        zip(
            classes,
            probabilities
        )
    )

    return (
        clean_text,
        prediction,
        probability_dict
    )


# ============================================
# STEP 8: INTERACTIVE PREDICTION SYSTEM
# ============================================

print("\n" + "=" * 50)
print("TEXT SENTIMENT PREDICTION SYSTEM")
print("=" * 50)

print("\nEnter a sentence to classify.")

print("\nType 'exit' to stop the program.")


while True:

    print("\n" + "-" * 50)

    text = input(
        "Enter text: "
    )


    # Exit condition
    if text.lower() == "exit":

        print("\nPrediction system stopped.")

        break


    # Empty input check
    if not text.strip():

        print("\nPlease enter some text.")

        continue


    # Make prediction
    clean_text, prediction, probabilities = predict_sentiment(
        text
    )


    # ========================================
    # DISPLAY RESULTS
    # ========================================

    print("\nCleaned Text:")
    print(clean_text)


    print("\nPrediction:")
    print(prediction.upper())


    print("\nConfidence:")

    confidence = probabilities[prediction] * 100

    print(
        f"{confidence:.2f}%"
    )


    print("\nClass Probabilities:")

    for label, probability in probabilities.items():

        print(
            f"{label}: "
            f"{probability * 100:.2f}%"
        )


# ============================================
# COMPLETION
# ============================================

print("\n" + "=" * 50)

print("STEP 13 COMPLETED")

print("=" * 50)