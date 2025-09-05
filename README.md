Heart Disease Prediction using Machine Learning
Overview

This project uses machine learning to predict the likelihood of heart disease in patients based on their medical data. It demonstrates key ML steps including data preprocessing, exploratory analysis, feature selection, and supervised learning. The goal is to help healthcare professionals identify at-risk patients for early intervention.

Features

Data preprocessing and cleaning

Exploratory Data Analysis (EDA) with visualizations

Feature selection for better model performance

Dimensionality reduction using PCA

Supervised learning models like Logistic Regression and Random Forest

Model evaluation using accuracy, precision, recall, and F1-score

Technologies Used

Python

Pandas & NumPy

Scikit-learn

Matplotlib & Seaborn

Installation



Navigate to the project folder:

cd heart-disease-ml


Install required libraries:

pip install -r requirements.txt

Usage

Open the Jupyter notebooks in the notebooks/ folder.

Follow the workflow:

01_data_preprocessing.ipynb → clean and prepare data

02_pca_analysis.ipynb → analyze and reduce dimensions

03_feature_selection.ipynb → select important features

04_supervised_learning.ipynb → train and evaluate models

05_unsupervised_learning.ipynb → optional clustering analysis         
Dataset

The project uses a standard heart disease dataset (included in data/heart_disease.csv). It contains features like age, sex, blood pressure, cholesterol, and other medical indicators.
