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
-  Before building the model, the dataset is splitted into 80% for training set and 20% for testing set to evaluate model performance.
-  Builded regression models to predict Revenue based on Campaign features.
-  Builded classification models to predict Profit/Loss using  New Calculated ROI feature.
-  Trained the models using appropriate algorithms such as Linear Regression, Logistic Regression, Decision Tree, Random Forest, Gradient Boost and AdaBoost.
-  Perform feature selection and ensure no data leakage.

# Model evaluation:
Model evaluation is the process of measuring how well the machine learning models perform on unseen data.

**Regression:**
- Regression is a supervised learning technique used to predict continuous numerical values by learning relationships between input variables (features) and an output variable (target).

**Classification:**
- Classification is also a supervised learning technique where an algorithm is trained with labeled data to predict the category of new data.

**The Models used in this project,**
- Linear Regression, Logistic Regression, Decision Tree, Random Forest, Gradient Boost and AdaBoost.

**Linear Regression:**
- Linear Regression is a fundamental supervised learning algorithm used to model the relationship between a dependent variable and one or more independent variables.
- Its main focus is to find linear relationship between Inputs and Outputs, where predicts continuous values by fitting a straight line that represents the data.

**Logistic Regression:**
- Logistic Regression is a supervised machine learning classification algorithm used to predict the probability of a categorical outcome (such as Yes/No, Profit/Loss, Spam/Not Spam) based on one or more input features.

**Decision tree:**
- A decision tree is a supervised learning algorithm used for both classification and regression tasks. It has a hierarchical tree structure which consists of a root node, branches, internal nodes and leaf nodes.
- It works like a flowchart that helps in making step-by-step decisions

**Random Forest**:
- Random Forest is an ensemble learning method that combines multiple decision trees to produce more accurate and stable predictions.
- It can be used for both classification and regression tasks

**Adaboost:**
- AdaBoost (Adaptive Boosting) is a machine learning boosting technique used as an ensemble method to adjust the weights of training samples and combine multiple weak classifiers into a single strong classifier.
- It is used for both Classification and regression models.

**Gradient Boost:**
- Gradient boosting is a machine learning technique that builds a highly accurate predictive model by combining multiple simple models (usually decision trees) one by one.
- Instead of making independent predictions, each new model is specifically trained to fix the errors and mistakes made by the models that came before.
- It is used  for both classification and regression tasks.

**Evaluate regression models using RMSE, MAE, R², MSE:**

**Mean Absolute Error (MAE):**
- Average of absolute differences between actual and predicted values.
- Difference between actual and predicted values is Residuals.
- <img width="396" height="287" alt="image" src="https://github.com/user-attachments/assets/9258db4b-fdcd-4b7b-9edc-794df9c80728" />



**Mean Square Error (MSE):**
- Average of squared differences between actual and predicted values.
- <img width="356" height="277" alt="image" src="https://github.com/user-attachments/assets/44e5d1d6-7bf4-449b-b7ea-a3364cd04dd4" />



**Root Mean Square Error (RMSE):**
- Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors.
<img width="398" height="295" alt="image" src="https://github.com/user-attachments/assets/1b136ddd-936a-4080-b5c5-c3158d2101b0" />



**R^2 Score:**
- R-squared represents the proportion of variance in the target variable that is explained by the regression model.
- It is a statistical measure that tells you how well a model explains and predicts future outcomes.
<img width="432" height="272" alt="image" src="https://github.com/user-attachments/assets/4510ab90-362b-43bd-a69b-348b9a859ca2" />



**Regression Models Performance:**
<img width="1068" height="145" alt="image" src="https://github.com/user-attachments/assets/9d4d20fa-113a-4fdd-b517-26990d90a61a" />


**Evaluate classification models using Accuracy, Precision, Recall, F1-score:**



- Compare model performance and select the best model.

# Techologies Used:
-  Python is used for programming, where data cleaning is done.
-  Pandas is a python library, used to read the raw data file and convert it into data frame for processing.
-  sklearn is for model selection, preprocessing, metrics, linearmodel for linear and logistic regression, tree for decision tree regressor and classifier, ensemble for Randomforest, Gradient Boosting and AdaBoost for both Regressor and Classifier.
-  imblearn for Over sampling to handle class imbalance.
-  joblib is used to dumb the model file as Pkl format.

# Dataset:
Source : Marketing Campaign Performance Prediction Datasets:
-  nykaa_campaign_data_with_nulls.csv
-  purplle_campaign_data_with_nulls.csv
-  tira_campaign_data_with_nulls.csv

# Project Structure:
<img width="577" height="286" alt="image" src="https://github.com/user-attachments/assets/667d68e9-62a4-4d73-a045-5a0b0737c8f9" />

















