# ✈️ Airline Reviews Sentiment Analysis

## 📌 Overview
This project automates sentiment classification of airline passenger reviews using a hybrid approach that combines **rule-based sentiment analysis** (VADER, TextBlob) with **machine learning ensemble models**.  
It helps airlines gain insights into customer experiences, classify sentiments as **Positive, Neutral, or Negative**, and predict whether a customer would recommend the airline.

---

## 📊 Dataset
- **Source**: [Kaggle - Airline Reviews Dataset](https://www.kaggle.com/code/muhammadadeelabid/eda-preprocessing-ensemble-learning-86)
- **Rows**: ~23,000 reviews  
- **Columns**: 20 features including review content, reviewer metadata, and ratings

### 🎯 Target Variables
- **Sentiment (derived)**: Positive / Neutral / Negative  
- **Recommendation (binary)**: Yes / No

---

## ⚙️ Workflow

### 📋 Exploratory Data Analysis (EDA)
- Handled missing values
- Visualized distributions of ratings and review content
- Generated word clouds for different sentiment classes

### 🧹 Text Preprocessing
- Lowercased text, removed punctuation and stopwords
- Tokenized and lemmatized text using **TextBlob**
- Vectorized with **TF-IDF**

### 🧠 Sentiment Labeling
Used a **hybrid ensemble of rule-based tools**:
- **VADER Review**: 0.45
- **VADER Title**: 0.25
- **TextBlob Review**: 0.20
- **TextBlob Title**: 0.10

> Final sentiment is determined by a **weighted average** of the compound/polarity scores.

### 🤖 Model Training
Supervised learning models used:
- Logistic Regression
- Support Vector Classifier (SVC - RBF kernel)
- Decision Tree Classifier
- Random Forest Classifier

All models are combined using a **VotingClassifier (soft voting)** for the final prediction.

### 📈 Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- Confusion Matrix and Classification Report

---

## 🧪 Results

### ✅ Ensemble Classifier Performance
- **Accuracy**: 85%
- **Precision**: 0.84
- **Recall**: 0.85
- **F1 Score**: 0.84

### 📊 Performance Summary

| Metric Type       | Precision | Recall | F1-Score |
|-------------------|-----------|--------|----------|
| Macro Average     | 0.57      | 0.57   | 0.56     |
| Weighted Average  | 0.84      | 0.85   | 0.84     |

- **Best Accuracy (Voting Classifier)**: ~86%
- Ensemble model outperformed individual classifiers
- High consistency in sentiment classification and recommendation prediction

---

## 🛠️ Tech Stack

### Language
- Python 3.6+

### Environment
- Jupyter Notebook

### Libraries
- **NLP**: `nltk`, `textblob`, `vaderSentiment`, `re`
- **ML**: `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`, `wordcloud`, `missingno`
- **Data Handling**: `pandas`, `numpy`
