\# 🤖 AI Question Answering System Using Transformer



A context-based Question Answering application built using a pre-trained HuggingFace Transformer model. The system accepts a user-provided context and question, then extracts the most relevant answer from the context along with a confidence score.



\## 📌 Project Overview



This project is part of \*\*Week 2 – Natural Language Processing (NLP), Task 03\*\* of the AI internship.



The application uses the \*\*DistilBERT Transformer model\*\* fine-tuned on the Stanford Question Answering Dataset (SQuAD) to perform extractive Question Answering.



Users can:



\* Enter any context or passage

\* Enter a question related to the context

\* Generate an answer using the Transformer model

\* View the model confidence score

\* View the original context



\## 🎯 Objective



To develop an intelligent Question Answering system that can understand a given context and automatically extract the most relevant answer to a user's question using a pre-trained Transformer model.



\## 🧠 Model



\*\*Model:\*\* `distilbert-base-cased-distilled-squad`



The model is a lightweight DistilBERT model fine-tuned for extractive Question Answering.



It identifies the most relevant span of text within the supplied context that answers the user's question.



\## 🔄 System Workflow



```text

User Context

&#x20;     ↓

User Question

&#x20;     ↓

Question Answering Pipeline

&#x20;     ↓

DistilBERT Transformer

&#x20;     ↓

Answer Span Extraction

&#x20;     ↓

Answer + Confidence Score

```



\## 🛠️ Technologies Used



| Technology               | Purpose                       |

| ------------------------ | ----------------------------- |

| Python                   | Core programming language     |

| HuggingFace Transformers | Pre-trained Transformer model |

| PyTorch                  | Deep Learning backend         |

| Streamlit                | Web application interface     |

| DistilBERT               | Question Answering model      |

| SQuAD                    | Model fine-tuning dataset     |



\## 📂 Project Structure



```text

task3\_qna\_transformer/

│

├── app.py

├── qna\_model.py

├── requirements.txt

└── README.md

```



\## ⚙️ Installation



\### 1. Clone the repository



```bash

git clone https://github.com/gayatrisaboji/AI-Internship-Projects.git

```



\### 2. Navigate to the project



```bash

cd AI-Internship-Projects/week2\_nlp/task3\_qna\_transformer

```



\### 3. Create a virtual environment



```bash

python -m venv qna\_venv

```



\### 4. Activate the virtual environment



\*\*Windows PowerShell:\*\*



```powershell

.\\qna\_venv\\Scripts\\Activate.ps1

```



\### 5. Install dependencies



```bash

pip install -r requirements.txt

```



\## ▶️ Running the Application



Start the Streamlit application:



```bash

streamlit run app.py

```



The application will open in your browser at:



```text

http://localhost:8501

```



\## 🧪 Example



\### Context



```text

Python is a high-level programming language created by

Guido van Rossum. It was first released in 1991.

Python is widely used for artificial intelligence,

machine learning, data science, and web development.

```



\### Question



```text

Who created Python?

```



\### Output



```text

Answer: Guido van Rossum

Confidence: Model dependent

```



\## 📊 Features



\### 1. Context-Based Question Answering



The application accepts arbitrary text as context instead of relying on a fixed dataset.



\### 2. Transformer-Based NLP



Uses a pre-trained DistilBERT Transformer model for extractive Question Answering.



\### 3. Confidence Score



Displays the model's confidence score for the extracted answer.



\### 4. Interactive Interface



Streamlit provides a simple and interactive web interface for entering contexts and questions.



\### 5. Answer Position



The underlying model also provides the start and end positions of the extracted answer within the context.



\## 🧪 Testing



The model was tested using different contexts and questions, including:



\* Artificial Intelligence

\* Python programming

\* Taj Mahal

\* Machine Learning



Example:



```text

Context:

The Taj Mahal is a white marble monument located in

Agra, India. It was commissioned by Mughal emperor

Shah Jahan in memory of his wife Mumtaz Mahal.



Question:

Who commissioned the Taj Mahal?



Answer:

Shah Jahan

```



\## 📈 Advantages



\* Uses a pre-trained Transformer model

\* No need to train a model from scratch

\* Supports user-provided contexts

\* Provides confidence scores

\* Simple and interactive Streamlit interface

\* Suitable for educational and NLP demonstrations



\## ⚠️ Limitations



\* The model can only answer questions based on the supplied context.

\* It may produce incorrect answers when the context does not contain the required information.

\* Confidence scores should not be interpreted as guaranteed accuracy.

\* Very long contexts may require additional processing or chunking.



\## 🚀 Future Improvements



Possible improvements include:



\* Support for PDF and document uploads

\* Multiple question answering

\* Context history

\* Improved answer visualization

\* Support for larger Transformer models

\* Answer highlighting inside the context

\* Question Answering analytics



\## 🎓 Internship Information



\*\*Week:\*\* 2

\*\*Domain:\*\* Natural Language Processing

\*\*Task:\*\* 03 – Question Answering Model Using Transformer



\## 👩‍💻 Author



\*\*Gayatri Saboji\*\*



B.E. – Information Science and Engineering



\## 📜 License



This project was developed for educational and internship purposes.



