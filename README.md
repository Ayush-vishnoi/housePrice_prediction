# 🏠 House Price Prediction using Machine Learning

### Live Demo-https://housepriceprediction-sqvlnhmncd5k39fbxoqqmb.streamlit.app
## Overview

This project builds an end-to-end Machine Learning pipeline to predict residential property prices using the Ames Housing Dataset.

The project covers the complete ML workflow including:

* Data Exploration & Cleaning
* Missing Value Handling
* Feature Engineering
* Model Training & Evaluation
* Feature Importance Analysis
* Residual Analysis
* Streamlit Deployment

The goal is not only to predict house prices accurately but also to understand which factors influence property values the most.

---

## Dataset

### Ames Housing Dataset

The dataset contains detailed information about residential properties sold in Ames, Iowa.

### Dataset Statistics

* Total Records: 2,930
* Features: 80+ Housing Attributes
* Target Variable: SalePrice

Features include:

* Property Size
* Lot Area
* Neighborhood Information
* Garage Details
* Basement Features
* Construction Quality
* Year Built
* Interior & Exterior Characteristics

---

## Project Workflow

### 1. Data Exploration

Performed:

* Dataset inspection
* Missing value analysis
* Summary statistics
* Target distribution analysis

### 2. Data Preprocessing

Implemented:

* Missing value imputation
* One-Hot Encoding for categorical variables
* Feature scaling
* Train/Test split

### 3. Feature Engineering

Created:

* Full feature dataset
* Lightweight feature subset using most important features

### 4. Model Training

Two models were developed:

#### Full Random Forest Model

Uses all available housing features.

#### Lite Random Forest Model

Uses only the 10 most important features:

* Overall Qual
* Gr Liv Area
* Garage Cars
* Garage Area
* 1st Flr SF
* Total Bsmt SF
* Year Built
* Lot Area
* Overall Cond
* Year Remod/Add

---

## Model Performance

| Model              |             Features | R² Score |
| ------------------ | -------------------: | -------: |
| Full Random Forest | 302 Encoded Features |    0.918 |
| Lite Random Forest |          10 Features |    0.904 |

### Key Observation

The Lite Model achieves nearly the same performance while using only 10 features, making it suitable for deployment and real-time predictions.

---

## Feature Importance Analysis

Top factors influencing house prices:

1. Overall Quality
2. Ground Living Area
3. Garage Capacity
4. Garage Area
5. First Floor Area
6. Basement Area
7. Year Built
8. Lot Area

Overall Quality was the strongest predictor of property value.

---

## Residual Analysis

Residual analysis showed:

* Errors centered around zero
* No significant prediction bias
* Good generalization on unseen data
* Few large prediction errors

This indicates that the model captures housing price patterns effectively.

---

## Streamlit Application

An interactive Streamlit web application was developed to allow users to:

* Enter house details
* Predict property prices instantly
* View model performance
* Explore feature importance and residual analysis

### Features

* User-friendly interface
* Real-time predictions
* Model comparison table
* Visualization support

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

---

## Project Structure

```text
house_prediction/
│
├── data/
│
├── models/
│   ├── full_rf_model.pkl
│   └── lite_rf_model.pkl
│
├── plots/
│   ├── feature_importance.png
│   └── residual_plot.png
│
├── app.py
├── notebook.ipynb
├── requirements.txt
└── README.md
```

---

## Results

* Full Model R²: 0.918
* Lite Model R²: 0.904
* High prediction accuracy
* Explainable machine learning approach
* Successful Streamlit deployment

---

## Future Improvements

* Hyperparameter tuning
* Cross-validation
* SHAP Explainability
* XGBoost implementation
* Cloud deployment

---

## Conclusion

This project demonstrates a complete machine learning workflow for real estate price prediction. By combining robust preprocessing, Random Forest Regression, feature importance analysis, and Streamlit deployment, the model achieves strong predictive performance while remaining interpretable and deployment-ready.
