\# Week 2 - Task 02: Resume Parser using NLP



\## 📌 Project Overview



The Resume NLP Parser is an AI/NLP-based application that automatically extracts useful information from resume PDF files.



The system extracts:



\- Personal Information

\- Skills

\- Education

\- Work Experience

\- Certifications

\- Projects

\- Languages

\- Named Entities



The extracted information is returned as structured JSON and can be accessed through a Flask REST API and a Streamlit web interface.



\---



\## 🛠️ Technologies Used



\- Python

\- NLP

\- spaCy

\- Flask

\- Streamlit

\- PDFPlumber

\- Regular Expressions

\- JSON

\- Requests



\---



\## 🏗️ Project Architecture



```text

Resume PDF

&#x20;   ↓

PDF Text Extraction

&#x20;   ↓

NLP Processing

&#x20;   ↓

Information Extraction

&#x20;   ↓

Named Entity Recognition

&#x20;   ↓

Structured JSON

&#x20;   ↓

Flask REST API

&#x20;   ↓

Streamlit Web Interface

