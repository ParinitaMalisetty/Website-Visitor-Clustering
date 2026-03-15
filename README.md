# Website Visitor Clustering (NLP-based Intent Detection)

Focuses on semantic intent detection and behavioral clustering of website visitors using Natural Language Processing (NLP).

## 📌 Project Overview
This system processes website interaction logs and search queries to automatically group visitors into distinct personas. Unlike traditional numeric clustering, this tool utilizes **TF-IDF Vectorization** and **Logistic Regression** to understand the linguistic intent behind user searches.

## 🚀 Key Features
* **Text Preprocessing**: Automated cleaning including tokenization, punctuation removal, and stopword filtering using NLTK.
* **Feature Extraction**: Converts raw text into numerical data using TF-IDF with bigrams to capture contextual phrases.
* **Intent Prediction**: Classifies users into segments (e.g., Transactional vs. Informational) with real-time confidence scoring.
* **Interactive UI**: A web-based dashboard built with **Streamlit** for real-time query testing.
* **Data Visualization**: Dynamic charts and statistical summaries powered by **Plotly** and **Matplotlib**.

## 🛠️ Tech Stack
* **Language**: Python 3.x
* **Framework**: Streamlit
* **NLP Library**: NLTK (Natural Language Toolkit)
* **Machine Learning**: Scikit-learn
* **Data Analysis**: Pandas & NumPy
* **Visualization**: Plotly, Matplotlib, & Seaborn

## 📋 System Requirements
* **IDE**: Visual Studio Code (VS Code) or Google Colab
* **Hardware**: Intel i3/i5, 4GB RAM (Min)
* **Environment**: Stable internet connection for NLTK dataset downloads

## 🔧 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone [https://github.com/ParinitaMalisetty/Website-Visitor-Clustering.git](https://github.com/ParinitaMalisetty/Website-Visitor-Clustering.git)
   cd Website-Visitor-Clustering
   ```

2. **Install Dependencies**
   ```bash
   pip install streamlit pandas numpy nltk scikit-learn plotly matplotlib seaborn
   ```

3. **Run The Application**
   ```bash
   streamlit run main.py / streamlit run original.py
   ```

## 📊 **Methodology**
The system follows a modular NLP and ML pipeline:
Data Preparation: Loading datasets (website_traffic.csv, test_data.csv) using Pandas.
Preprocessing: Lowercasing, tokenization, and removing non-essential stopwords while preserving signal words like "buy" or "price".
Vectorization: Transforming cleaned text into feature matrices using TfidfVectorizer.
Classification: Training a Logistic Regression model to distinguish between intent labels.
Evaluation: Utilizing cross-validation and accuracy metrics to ensure model reliability.
