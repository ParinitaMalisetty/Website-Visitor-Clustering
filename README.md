# Website Visitor Clustering using NLP

A micro-project developed for the **Artificial Intelligence and Machine Learning Lab**, focusing on semantic intent detection and behavioral clustering of website visitors using Natural Language Processing (NLP).

## 📌 Project Overview
This project processes website interaction logs and search queries to automatically group visitors into distinct personas. [cite_start]Unlike traditional numeric clustering, this system utilizes **TF-IDF Vectorization** and **Logistic Regression** to understand the linguistic intent behind user searches [cite: 58-60, 460].

## 🚀 Key Features
* [cite_start]**Text Preprocessing**: Automated cleaning including tokenization, punctuation removal, and stopword filtering using NLTK [cite: 86-91, 131-135].
* [cite_start]**Feature Extraction**: Converts raw text into numerical data using TF-IDF with bigrams to capture contextual phrases [cite: 94-95, 335].
* [cite_start]**Intent Prediction**: Classifies users into segments (e.g., Transactional vs. Informational) with real-time confidence scoring [cite: 100-101, 388-394].
* [cite_start]**Interactive UI**: A web-based dashboard built with Streamlit for real-time query testing[cite: 161, 418].
* [cite_start]**Data Visualization**: Dynamic charts and statistical summaries powered by Plotly and Matplotlib [cite: 148-151, 153].

## 🛠️ Tech Stack
* [cite_start]**Language**: Python 3.8+ [cite: 115]
* [cite_start]**Framework**: Streamlit [cite: 116]
* [cite_start]**NLP Library**: NLTK (Natural Language Toolkit) [cite: 119]
* [cite_start]**Machine Learning**: Scikit-learn [cite: 120, 157]
* [cite_start]**Data Analysis**: Pandas & NumPy [cite: 122]
* [cite_start]**Visualization**: Plotly, Matplotlib, & Seaborn [cite: 120-121]

## 📋 System Requirements
* **IDE**: Visual Studio Code (VS Code)
* [cite_start]**Hardware**: Intel i3/i5, 4GB RAM (Min) [cite: 111-112]
* [cite_start]**Environment**: Stable internet connection for NLTK dataset downloads [cite: 113, 191-193]

## 🔧 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone [https://github.com/ParinitaMalisetty/Website-Visitor-Clustering.git](https://github.com/ParinitaMalisetty/Website-Visitor-Clustering.git)
   cd Website-Visitor-Clustering
