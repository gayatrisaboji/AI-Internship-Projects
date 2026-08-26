# ============================================
# WEEK 2 - TASK 01
# TEXT CLASSIFICATION
# STEP 14 - STREAMLIT WEB APPLICATION
# ============================================

import os
import joblib
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="AI Sentiment Classifier",
    page_icon="🤖",
    layout="centered"
)


# ============================================
# NLTK RESOURCES
# ============================================

nltk.download(
    "punkt",
    quiet=True
)

nltk.download(
    "punkt_tab",
    quiet=True
)

nltk.download(
    "stopwords",
    quiet=True
)


# ============================================
# FILE PATHS
# ============================================

MODEL_PATH = "models/text_classifier_model.pkl"

VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


# ============================================
# CHECK MODEL FILES
# ============================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "Classification model not found. "
        "Please run train.py first."
    )

    st.stop()


if not os.path.exists(VECTORIZER_PATH):

    st.error(
        "TF-IDF vectorizer not found. "
        "Please run train.py first."
    )

    st.stop()


# ============================================
# LOAD MODEL AND VECTORIZER
# ============================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    return model, vectorizer


model, vectorizer = load_model()


# ============================================
# STOPWORDS
# ============================================

stop_words = set(
    stopwords.words("english")
)


# ============================================
# TEXT PREPROCESSING
# ============================================

def preprocess_text(text):

    text = text.lower()

    tokens = word_tokenize(text)

    tokens = [

        word

        for word in tokens

        if word.isalnum()
        and word not in stop_words

    ]

    return " ".join(tokens)


# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_sentiment(text):

    clean_text = preprocess_text(
        text
    )

    text_tfidf = vectorizer.transform(
        [clean_text]
    )

    prediction = model.predict(
        text_tfidf
    )[0]

    probabilities = model.predict_proba(
        text_tfidf
    )[0]

    probability_dict = dict(
        zip(
            model.classes_,
            probabilities
        )
    )

    return (
        clean_text,
        prediction,
        probability_dict
    )


# ============================================
# APPLICATION HEADER
# ============================================

st.title(
    "🤖 AI Sentiment Classifier"
)

st.subheader(
    "NLP Text Classification using Machine Learning"
)


st.write(
    "Enter a sentence below and the trained "
    "Logistic Regression model will classify "
    "its sentiment."
)


# ============================================
# TEXT INPUT
# ============================================

text = st.text_area(

    "Enter your text:",

    placeholder=(
        "Example: "
        "I really love this product!"
    ),

    height=150
)


# ============================================
# PREDICTION BUTTON
# ============================================

if st.button(
    "🔍 Predict Sentiment",
    use_container_width=True
):

    if not text.strip():

        st.warning(
            "Please enter some text first."
        )

    else:

        clean_text, prediction, probabilities = (
            predict_sentiment(text)
        )


        # ====================================
        # DISPLAY CLEANED TEXT
        # ====================================

        st.subheader(
            "Preprocessed Text"
        )

        st.code(
            clean_text
        )


        # ====================================
        # DISPLAY PREDICTION
        # ====================================

        st.subheader(
            "Prediction"
        )


        confidence = (
            probabilities[prediction] * 100
        )


        if prediction == "positive":

            st.success(
                f"😊 POSITIVE"
            )

        else:

            st.error(
                f"😞 NEGATIVE"
            )


        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        # ====================================
        # PROBABILITIES
        # ====================================

        st.subheader(
            "Class Probabilities"
        )


        negative_probability = (
            probabilities.get(
                "negative",
                0
            ) * 100
        )


        positive_probability = (
            probabilities.get(
                "positive",
                0
            ) * 100
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Negative",
                f"{negative_probability:.2f}%"
            )


        with col2:

            st.metric(
                "Positive",
                f"{positive_probability:.2f}%"
            )


        # ====================================
        # PROBABILITY BAR
        # ====================================

        st.write(
            "Positive Probability"
        )


        st.progress(
            int(positive_probability)
        )


# ============================================
# SIDEBAR
# ============================================

with st.sidebar:

    st.header(
        "About the Project"
    )


    st.write(
        """
        This application performs sentiment
        classification using Natural Language
        Processing and Machine Learning.
        """
    )


    st.write(
        "**Technologies:**"
    )


    st.write(
        """
        • Python

        • NLTK

        • TF-IDF

        • Scikit-learn

        • Logistic Regression

        • Streamlit
        """
    )


    st.write(
        "**Model:**"
    )


    st.write(
        "Logistic Regression"
    )


    st.write(
        "**Classes:**"
    )


    st.write(
        "Positive / Negative"
    )


# ============================================
# FOOTER
# ============================================

st.divider()


st.caption(
    "Week 2 NLP Internship Project — "
    "Text Classification"
)