# naanmudhalvan_genAI
# Emotion Detection 

## Overview
This project aims to classify emotions based on textual data using machine learning techniques. It involves preprocessing text data, training various classifiers, and evaluating their performance.

## Abstract
This project aims to develop a robust machine learning model for emotion detection based on textual data. Leveraging natural language processing techniques, the dataset is preprocessed to clean, tokenize, and convert text into numerical features using TF-IDF vectorization. Various machine learning algorithms are trained on the preprocessed data, including Random Forest Classifier, Support Vector Machine, Multinomial Naive Bayes Classifier, Gradient Boosting Classifier, and Logistic Regression Classifier. Evaluation metrics such as accuracy, precision, recall, and F1-score are utilized to assess model performance. Additionally, exploratory data analysis techniques are employed to gain insights into the dataset, with visualizations such as word clouds and TF-IDF score distributions aiding in understanding the text data's structure and characteristics. Through this project, the goal is to provide a reliable solution for emotion classification, with potential applications in sentiment analysis, customer feedback analysis, and mental health assessment.

## Dataset
The dataset used in this project contains textual data along with corresponding emotion labels. It consists of 839555 samples with 6 features. Download the dataset from https://www.kaggle.com/datasets/simaanjali/emotion-analysis-based-on-text 

![image](https://github.com/madulika-prabu/naanmudhalvan_genAI/assets/131234604/957a5396-5e89-4ff8-a703-22a77bc5c92c)

![image](https://github.com/madulika-prabu/naanmudhalvan_genAI/assets/131234604/45d2471a-26b3-4492-8868-9e5318978c02)



## Features
- **Text**: Input textual data to be classified.
- **Emotion**: Target emotion label to be predicted.Emotion labels seem to be categorical values representing different emotions (e.g., hate, neutral, anger, love, worry).
- **Unnamed: 0**: This column seems to be an index or identifier for each row.
- **text_length**: This column might represent the length of the text in terms of characters or words.
- **clean_text**: This column likely contains the preprocessed or cleaned version of the text data.

## Preprocessing
- Text cleaning: Remove noise, punctuation, and special characters.
- Tokenization: Convert text into tokens (words or phrases).
- TF-IDF vectorization: Convert tokens into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).

## Models Trained
- **Random Forest Classifier**: A decision tree-based ensemble learning algorithm.
- **Support Vector Machine (SVM) Classifier**: A supervised learning algorithm for classification tasks.
- **Multinomial Naive Bayes Classifier**: A probabilistic classifier based on Bayes' theorem.
- **Gradient Boosting Classifier**: A boosting ensemble learning algorithm that builds trees sequentially.
- **Logistic Regression Classifier**: A linear model for binary classification tasks.

## Evaluation Metrics
- **Accuracy**: Overall classification accuracy.
- **Classification Report**: Precision, recall, F1-score, and support for each class.

## Usage
1. Install the required dependencies listed in `requirements.txt`.
2. Run `preprocess.py` to preprocess the dataset and generate TF-IDF features.
3. Run `train.py` to train the classifiers on the preprocessed data.
4. Evaluate the trained models using `evaluate.py`.

## Requirements
- Python 3.x
- scikit-learn
- pandas
- numpy

## Future Work
- Experiment with deep learning models like LSTM for better performance.
- Explore advanced text preprocessing techniques.
- Optimize hyperparameters and model architectures for improved accuracy.
