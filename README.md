# Project Title:
Multi-Brand Marketing Campaign Performance Analysis and Prediction

# Project Overview:
The project aim is to focus on analyze and predict the performance of marketing campaigns across multiple brands such as Nykaa, Purplle, and Tira using Machine Learning.

# Objectives:
The objective of this project is to design and implement a Marketing Campaign Performance Prediction System that follows the complete machine learning lifecycle:
- Data collection
- Data Cleaning & Preprocessing
- Exploratory data Analysis (EDA)
- Feature Engineering
- Model Building
- Model Evaluation

# Data Collection:
Data collection is to gather and organize the raw data in a structured format, making it ready for preprocessing, analysis, feature engineering, and machine learning model development.

# Data Cleaning & Preprocessing:
Data preprocessing is the process of cleaning and preparing raw data so that it can be effectively used for analysis and machine learning.
- Handle missing values (null data) to ensures the dataset is complete and prevents errors during analysis and model training.
- Remove duplicate records to prevents the same data from being counted multiple times, which could change the analysis and predictions.
- Perform data type conversion to enable accurate calculations, filtering, and machine learning operations.
- Clean and standardize the dataset to improve data consistency and reduces errors during analysis.

# Exploratory Data Analysis (EDA):
Exploratory Data Analysis (EDA) is used to summarize, visualize, and understand the dataset. It helps identify patterns, trends, and relationships among variables that influence campaign performance.
- Analyze campaign performance across each brands to identify the strongest and weakest performing brands.
- Identifying top-performing and low-performing campaigns which helps to understand successful campaign strategies and avoid ineffective ones.
- Correlation analysis is to measure the relationship between different marketing campaign variables.
- Analyze marketing channels which helps to allocate marketing budgets to the most effective channels.

# Feature Engineering:
Feature Engineering is the process of creating and transforming variables to improve the performance of machine learning models
- Applying One-hot Encoding technique for Channel_Used column which has Email, Facebook, Google, Instagram, WhatsApp and YouTube.
- Applying Label Encoding technique for Campaign, Campaign_Type, Customer_Segment, Target_Audience and Language where text format is converted into numbers.
- Create Profit_Loss feature which serves as the target variable for the classification model, helping predict whether a campaign will be profitable or not.

# Model Building:
Model Building is the Process of training machine learning algorithms using the prepared dataset to predict campaign performance
-  Before building the model, the dataset is splitted into 80% for traing set and 20% for testing set to evaluate model performance.
-  Builded regression models to predict Revenue based on Campaign features.
-  Builded classification models to predict Profit/Loss using  New Calculated ROI feature.
-  Trained the models using appropriate algorithms such as Linear Regression, Logistic Regression, Decision Tree, Random Forest, Gradient Boost and AdaBoost.
-  Perform feature selection and ensure no data leakage.

# Model evaluation:
Model evaluation is the process of measuring how well the machine learning models perform on unseen data.
- Evaluate regression models using RMSE, MAE, R², MSE.
- Evaluate classification models using Accuracy, Precision, Recall, F1-score
- Compare model performance and select the best model.

# Techologies Used:
-  Python is used for programming, where data cleaning is done.
-  Pandas is a python library, used to read the raw data file and convert it into data frame for processing.
-  sklearn is for model selection, preprocessing, metrics, linearmodel for linear and logistic regression, tree for decision tree regressor and classifier, ensemble for Randomforest, Gradient Boosting and AdaBoost for both Regressor and Classifier.
-  imblearn for Over sampling to handle class imbalance.
-  joblib is used to dumb the model file as Pkl format.

# Dataset:
-  Source : Marketing Campaign Performance Prediction Datasets:
-  Dataset Files
-  nykaa_campaign_data_with_nulls.csv
-  purplle_campaign_data_with_nulls.csv
-  tira_campaign_data_with_nulls.csv
















