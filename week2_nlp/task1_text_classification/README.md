# 🤖 AI Sentiment Classification using NLP

## 📌 Project Overview

This project implements a Natural Language Processing (NLP) based text classification system that classifies text into two sentiment categories:

- Positive
- Negative

The project uses text preprocessing with NLTK, TF-IDF for feature extraction, and Logistic Regression for machine learning classification.

A Streamlit web application is also provided for real-time sentiment prediction.

---

# 🎯 Objectives

The main objectives of this project are:

- Understand basic Natural Language Processing
- Preprocess text using NLTK
- Convert text into numerical features using TF-IDF
- Train a machine learning classification model
- Evaluate model performance
- Build a real-time prediction system
- Create a simple web interface using Streamlit

---

# 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NLTK | Text preprocessing |
| Pandas | Dataset handling |
| NumPy | Numerical processing |
| Scikit-learn | Machine learning |
| TF-IDF | Text feature extraction |
| Logistic Regression | Classification |
| Matplotlib | Data visualization |
| Joblib | Model serialization |
| Streamlit | Web application |

---

# 📂 Project Structure

```text
task1_text_classification/
│
├── data/
│   └── data.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── text_classifier_model.pkl
│
├── outputs/
│   ├── test_predictions.csv
│   ├── evaluation_results.csv
│   ├── confusion_matrix.png
│   └── model_metrics.png
│
├── train.py
├── visualize.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md