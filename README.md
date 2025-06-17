✈️ Airline Reviews Sentiment Analysis
📌 Overview
This project aims to automate the sentiment classification of airline passenger reviews using a hybrid approach combining rule-based sentiment analysis (VADER, TextBlob) with machine learning ensemble models. The output helps airlines gain insights into customer experiences, classify sentiments as positive, neutral, or negative, and predict whether a customer would recommend the airline.

📊 Dataset
Source: Kaggle Airline Reviews Dataset (https://www.kaggle.com/code/muhammadadeelabid/eda-preprocessing-ensemble-learning-86 )

Rows: ~23,000 reviews

Columns: 20 features including review content, reviewer metadata, and ratings

Target Variables:

Sentiment (derived): Positive / Neutral / Negative
Recommendation (binary): Yes / No
⚙️ Workflow
📋 Exploratory Data Analysis (EDA)

Handled missing values, visualized distributions, and examined review patterns
Word clouds generated for different sentiment classes
🧹 Text Preprocessing

Lowercasing, punctuation removal, stopword removal
Tokenization and lemmatization (TextBlob)
TF-IDF vectorization
🧠 Sentiment Labeling

Rule-based tools: VADER & TextBlob
Weighted Ensemble Scoring:
VADER Review: 0.45
VADER Title: 0.25
TextBlob Review: 0.20
TextBlob Title: 0.10
Final sentiment determined by weighted average of compound/polarity scores
🤖 Model Training

Trained the following supervised models:
Logistic Regression
Support Vector Classifier (SVC - RBF kernel)
Decision Tree Classifier
Random Forest Classifier
Combined using VotingClassifier (soft voting) for final prediction
📈 Evaluation

Metrics: Accuracy, Precision, Recall, F1-Score
Confusion Matrix and Classification Report
🧪 Results
Ensemble Classifier Evaluation: Current Performance Metrics
Accuracy: 85%
Precision: 0.84
Recall: 0.85
F1 Score: 0.84
Overall Performance:
Macro Average:
Precision: 0.57
Recall: 0.57
F1-Score: 0.56
Weighted Average:
Precision: 0.84
Recall: 0.85
F1-Score: 0.84
Best Accuracy (Voting Classifier): ~86%
The ensemble model outperformed individual classifiers
High consistency in both sentiment classification and recommendation prediction
🛠️ Tech Stack
Language: Python 3.6+
Environment: Jupyter Notebook
Libraries:

NLP: nltk, textblob, vaderSentiment, re
ML: scikit-learn
Visualization: matplotlib, seaborn, wordcloud, missingno
Data Handling: pandas, numpy
