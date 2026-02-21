# 📰 Fake News Detector using Machine Learning

This project is a Fake News Detection system built using **Machine Learning** and **Natural Language Processing (NLP)**.  
It classifies news articles as **Fake**, **Real**, or **Uncertain** based on linguistic patterns learned from historical data.

The system also provides **confidence percentages** to make predictions more transparent and interpretable.

---

## 📌 Project Description

With the rapid growth of digital media, misinformation spreads quickly across online platforms.  
Many users find it difficult to identify whether a news article is reliable or misleading.

This project aims to assist users by analyzing the textual content of news articles and predicting whether the news is fake or real using machine learning techniques.

---

## 🎯 Objectives

- Detect fake and real news articles using machine learning  
- Apply NLP techniques for text preprocessing  
- Provide confidence-based predictions instead of hard decisions  
- Build a simple and user-friendly interface for non-technical users  

---

## 🗂 Dataset Used

- **Dataset Name:** WELFake Dataset  
- **Labels:**
  - `0` → Real News  
  - `1` → Fake News  
- The dataset contains news titles and article content.
- Title and text are combined to improve prediction accuracy.

---

## ⚙️ Technologies & Tools Used

### Programming Language
- Python

### Libraries
- Pandas  
- NumPy  
- Scikit-learn  
- TF-IDF Vectorizer  
- Logistic Regression  
- Streamlit  
- Pickle  

---

## 🧠 Model Details

- **Feature Extraction:** TF-IDF (Term Frequency–Inverse Document Frequency)
- **Machine Learning Algorithm:** Logistic Regression
- **Data Split:** Stratified Train-Test Split
- **Accuracy Achieved:** ~95% on test dataset

---

## 🔍 Prediction Logic

The system uses confidence-based thresholding:

- **Real News:** Real confidence ≥ 55% and higher than fake confidence  
- **Fake News:** Fake confidence ≥ 60%  
- **Uncertain:** All other cases  

This approach avoids unsafe predictions and improves reliability.

---

## 🖥️ Web Interface

A web interface is developed using **Streamlit**, allowing users to:
- Paste a news article
- Get Fake / Real / Uncertain prediction
- View confidence percentages

No coding knowledge is required to use the application.

---

## 🚀 How to Run the Project

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
