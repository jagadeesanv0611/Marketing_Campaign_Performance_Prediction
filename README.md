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

****Overall Revenue by Each Brand:****

<img width="1342" height="575" alt="image" src="https://github.com/user-attachments/assets/287d34b5-3633-41b3-964d-e0fcfa773e85" />

****Overall ROI by each Campaign type:****

<img width="1227" height="697" alt="image" src="https://github.com/user-attachments/assets/99a0eb1a-672f-4f43-9d5b-ef7b15de4f09" />


****Overall Profit vs Loss:****

<img width="1353" height="595" alt="image" src="https://github.com/user-attachments/assets/0a257b99-de4d-4058-9e89-d31a32e02100" />

****Correlation Analysis:****

<img width="1052" height="900" alt="newplot (5)" src="https://github.com/user-attachments/assets/2ab1745f-fd52-4889-a7b9-3e033ac1621f" />


# Feature Engineering:
Feature Engineering is the process of creating and transforming variables to improve the performance of machine learning models
- Applying One-hot Encoding technique for Channel_Used column which has Email, Facebook, Google, Instagram, WhatsApp and YouTube.

- Applying Label Encoding technique for Campaign, Campaign_Type, Customer_Segment, Target_Audience and Language where text format is converted into numbers.

- CAMPAIGN_MAP        = {"Nykaa": 0, "Purplle": 1, "Tira": 2}

- CAMPAIGN_TYPE_MAP   = {"Email": 0,"Influencer": 1,"Paid Ads": 2,"SEO": 3, "Social Media": 4}

- TARGET_AUDIENCE_MAP = {"College Students": 0, "Premium Shoppers": 1, "Tier 2 City Customers": 2, "Working Women": 3,"Youth": 4}

- LANGUAGE_MAP        = {"Bengali": 0, "English": 1, "Hindi": 2, "Tamil": 3}

- CUSTOMER_SEGMENT_MAP = {"College Students": 0, "Premium Shoppers": 1, "Tier 2 City Customers": 2, "Working Women": 3, "Youth": 4}
  
- Create Profit_Loss feature which serves as the target variable for the classification model, helping predict whether a campaign will be profitable or not.

# Model Building:
Model Building is the Process of training machine learning algorithms using the prepared dataset to predict campaign performance
-  Before building the model, the dataset is splitted into 80% for training set and 20% for testing set to evaluate model performance.
-  X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X, y, test_size=0.2, random_state = 42)

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

****Mean Absolute Error (MAE):****
- Average of absolute differences between actual and predicted values.
- Difference between actual and predicted values is Residuals.
  <img width="396" height="287" alt="image" src="https://github.com/user-attachments/assets/9258db4b-fdcd-4b7b-9edc-794df9c80728" />


****Mean Square Error (MSE):****
- Average of squared differences between actual and predicted values.
  <img width="356" height="277" alt="image" src="https://github.com/user-attachments/assets/44e5d1d6-7bf4-449b-b7ea-a3364cd04dd4" />


****Root Mean Square Error (RMSE):****
- Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors.
  <img width="398" height="295" alt="image" src="https://github.com/user-attachments/assets/1b136ddd-936a-4080-b5c5-c3158d2101b0" />



****R^2 Score:****
- R-squared represents the proportion of variance in the target variable that is explained by the regression model.
- It is a statistical measure that tells you how well a model explains and predicts future outcomes.
  <img width="432" height="272" alt="image" src="https://github.com/user-attachments/assets/4510ab90-362b-43bd-a69b-348b9a859ca2" />


****Regression Models Performance:****
  <img width="1068" height="145" alt="image" src="https://github.com/user-attachments/assets/9d4d20fa-113a-4fdd-b517-26990d90a61a" />


****Model Comparison with R2 Score:****
<img width="681" height="596" alt="image" src="https://github.com/user-attachments/assets/6d1c1af8-3e8d-494c-af2c-d5606ea68667" />


****Random Forest Actual vs Predicted Revenue:****
  <img width="1331" height="710" alt="image" src="https://github.com/user-attachments/assets/43a88e8a-a33d-4b45-8702-e5e7b3fbb03a" />


**Evaluate classification models using Accuracy, Precision, Recall, F1-score:**
**Accuracy:**
- Accuracy is a fundamental metric used for evaluating the performance of a classification model.
- It is the proportion of correct predictions made by the model out of all predictions.
  <img width="460" height="86" alt="image" src="https://github.com/user-attachments/assets/fec41f81-67af-4a4d-9062-b0efc459fcc1" />
  <img width="362" height="77" alt="image" src="https://github.com/user-attachments/assets/f57974bc-e025-46e1-a704-c32d7c577bb6" />

**Precision:**
- Precision measures the proportion of true positive instances out of all predicted positive instances.
- It is calculated as the number of true positive instances divided by the sum of true positive and false positive instances.
  <img width="250" height="86" alt="image" src="https://github.com/user-attachments/assets/97247e38-9035-4d1e-9730-9a72c19a3f32" />

**Recall:**
- Recall or Sensitivity measures the proportion of true positive instances out of all actual positive instances.
- It is calculated as the number of true positive instances divided by the sum of true positive and false negative instances.
  <img width="210" height="72" alt="image" src="https://github.com/user-attachments/assets/df3d4b88-c9af-4c79-a2d4-1a5ed3c8640e" />

**F1 Score:**
- The F1 Score is the harmonic mean of precision and recall.
- It is useful when we need a balance between precision and recall as it combines both into a single number.
- A high F1 score means the model performs well on both metrics.
- Its range is [0,1].
  <img width="367" height="77" alt="image" src="https://github.com/user-attachments/assets/b4e4fd6f-ad08-4b38-bb4d-4449c45d42f0" />

**AUC and ROC Curve:**
- The ROC (Receiver Operating Characteristic) Area Under the Curve(AUC) score is a measure of the ability of a classifier to distinguish between positive and negative instances.
- It is calculated by plotting the true positive rate against the false positive rate at different classification thresholds and calculating the area under the curve.
  <img width="192" height="73" alt="image" src="https://github.com/user-attachments/assets/7fed2759-8d7c-4955-aa50-032f89fa8058" />
  <img width="195" height="78" alt="image" src="https://github.com/user-attachments/assets/5bd944d6-5b6f-48ba-8be1-acda090acaaa" />
  
  <img width="442" height="157" alt="image" src="https://github.com/user-attachments/assets/7fddbb49-3297-432b-852b-9818e80a7a3a" />


**Classification Model Performance:**

  <img width="1719" height="157" alt="image" src="https://github.com/user-attachments/assets/89830899-1e63-4cbd-8f14-7aa4cca45512" />


****Model Comparison of Classification:****

  <img width="675" height="602" alt="image" src="https://github.com/user-attachments/assets/9feba31d-69fd-4ce0-9a10-9f2f50ed0293" />


****Confusion Matrix:****

   <img width="1322" height="603" alt="image" src="https://github.com/user-attachments/assets/170d547b-51f3-427e-ba83-c9762766f49e" />


















