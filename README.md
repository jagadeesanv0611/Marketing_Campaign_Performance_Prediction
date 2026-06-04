# Project Title:
Multi-Brand Marketing Campaign Performance Analysis and Prediction

# Project Overview:
The Project aim is to predict **Revenue** and **Profit / Loss** for marketing campaigns using Machine Learning.

# Objectives:
The objective of this project is to design and implement a Marketing Campaign Performance Prediction System that follows the complete machine learning lifecycle:
1.	Data Collection:
     -   Loading the datasets into python using pandas Dataframe.
2.	Data Cleaning & Preprocessing:
     -   Handle missing values (null data)
     -	 Remove duplicate records
     -	 Perform data type conversion
     -	 Clean and standardize the dataset
     -	 Validate and correct ROI values

3.	Exploratory Data Analysis (EDA):
      -   By Analyzing campaign performance across each brands.
      -	  Identifying  top-performing and low-performing campaigns.
      -	  Explore relationships between Impressions, Leads, Spend, Clicks, Acquistion Cost, Revenue and New Calculated ROI using Coorrelation analysis.  
      -	  Analyzing channel-wise effectiveness

4.	Feature Engineering:
      -   Created a new feature Profit/Loss flag based on New calculated ROI.
      -   Applying One-hot Encoding technique for Channel_Used column which has Email, Facebook, Google, Instagram, WhatsApp and YouTube.
      -   Applying Label Encoding technique for Campaign, Campaign_Type, Customer_Segment, Target_Audience and Language where text format is converted into numbers.
  
5. Model Building:
       -  Before building the model, the dataset is splitted into traing set and testing sets to evaluate model performance.
       -  Builded regression models for predicting Revenue based on Campaign features.
       -  Builded classification models to predict Profit/Loss using  New Calculated ROI feature.
       -  Trained the models using appropriate algorithms such as Linear Regression, Logistic Regression, Decision Tree, Random Forest, Gradient Boost and AdaBoost.
       -  
















