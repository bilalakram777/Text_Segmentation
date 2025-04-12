
                                                                                 **Text Sentiment Analysis**
                                                                                          Overview

This project implements a sentiment analysis model using Natural Language Processing (NLP) techniques to classify movie reviews as either positive or negative. The model is built using Python and leverages libraries such as Pandas, NumPy, Scikit-learn, and NLTK.

**Table of Contents**
1 Installation
2 Dataset
3 Steps to Run the Code
4 Model Training and Evaluation
5 Sentiment Prediction
Observations

**1: Installation**

Required Libraries
You will need the following libraries:

pandas
numpy
scikit-learn
nltk
transformers (for advanced sentiment analysis)

NLTK Data
Make sure to download the necessary NLTK datasets by running the following commands in your Python environment:
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

**2:Dataset**
The dataset used in this project is imdb_reviews.csv, which contains movie reviews along with their corresponding sentiment labels (positive or negative). Ensure that this CSV file is in the same directory as your script.

**3: Steps to Run the Code**
Import Libraries: Import necessary libraries for data manipulation, machine learning, and text processing.
Load the Dataset: Load the dataset into a Pandas DataFrame.
Preprocess the Text: Clean the reviews by removing HTML tags, special characters, and stopwords, and perform lemmatization.
Split the Data: Split the dataset into training and testing sets using an 80-20 split.
TF-IDF Vectorization: Transform the text data into TF-IDF features.

**4: Train and Evaluate Model**
Train the Model: Train both Multinomial Naive Bayes and Logistic Regression models.
Evaluate the Model: Evaluate models using accuracy scores and classification reports.
Predict Sentiment: Define a function to predict sentiment for new reviews.

**5: Running the Scripts**
To run the scripts, execute the Python file in your terminal:
python your_script_name.py

**6: Observations**
The Multinomial Naive Bayes model achieved an accuracy of approximately 89.45%, effectively classifying sentiment.
The Logistic Regression model also performed well, providing a robust alternative.
Preprocessing steps significantly improved input data quality, enhancing model performance.
The ability to predict sentiment for new reviews demonstrates practical applications in movie recommendation systems and customer feedback analysis.

