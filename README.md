# Student Dropout Prediction
![Python](https://img.shields.io/badge/Python-3.8.10-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-LightGBM-orange)
![Frontend](https://img.shields.io/badge/Framework-Flask-red)
![Deployment](https://img.shields.io/badge/Cloud-Render-purple)

## Web Application: 
Built a web application using Flask and deployed on Render.

<img width="800" alt="image" src="doc/demo.gif">

## Check it out
https://student-dropout-prediction-1.onrender.com

## Introduction
Student dropout is a serious problem in the education system, where many students leave their courses before completing their studies due to academic, financial, or personal reasons. This project uses Machine Learning to predict students who are at risk of dropping out by analyzing historical and behavioral data. The system helps institutions take early action and provide timely support, improving student retention and overall educational outcomes.

## Problem Statement: 
This project aims to develop a Machine Learning model that predicts the probability of dropout rate from student data with three target outcomes 
* Dropout
* Enrolled
* Graduate

based on the other 34 features.

## Description: 
This data set is [predict-students-dropout-and-academic-success](https://www.kaggle.com/datasets/naveenkumar20bps1137/predict-students-dropout-and-academic-success) from Kaggle. The data set has been created from a higher education institution (acquired from several disjoint databases) related to students enrolled in different undergraduate degrees, such as agronomy, design, education, nursing, journalism, management, social service, and technologies. The dataset includes information known at the time of student enrollment (academic path, demographics, and social-economic factors) and the students' academic performance at the end of the first and second semesters.


### Exploratory Data Analysis:
* Exploratory Data Analysis is the first step of understanding your data and acquiring domain knowledge.


### Data Preprocessing and Feature Selection:


 On using **Correlation** method, I found that some features were moderately correlated so I removed the features with collinearity
 Also,Curriculum marks based features are correlated so i apply **PCA(Principal Component Analysis)** and transform all the curriculum features into one feature to reduce the dimensionality. Finally, we use 17 feature for training process.

 ### Model Training:
* On training my model using several classification algorithms such as *XGBoost*, *Random Forest*, *Extra Trees*, *Decision Trees*, *Light GBM* the model trained with **Light GBM** gave good results and less memory. 
* Also,used hyper-parameter tuning on Light GBM Classifier (baseline model) using **GridSearchCV**.
.
* As per the problem statement I used **Recall Score** as the evaluation metric for my model.


## Installation

* Clone this repository and check the ```requirements.txt```:
    ```shell
    git clone https://github.com/Dhruv-patel-17/Student_dropout_prediction
    cd Student_dropout_prediction
    pip install -r requirements.txt
    ```
* Simply run:    
    ```shell
    python app.py
    ```
