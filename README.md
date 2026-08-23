# 🚦 Traffic Demand Prediction using Stacked Ensemble Learning

A machine learning solution developed for **Flipkart Grid-Lock 2.0** to predict traffic demand using advanced feature engineering and ensemble learning techniques. The project combines multiple tree-based models through stacking to improve prediction accuracy and generalization.

## 🚀 Features

- Advanced temporal and spatial feature engineering
- Geohash-based location encoding
- Target encoding and label encoding
- Cyclical encoding for time-based features
- Stacked ensemble learning with multiple regressors
- 5-Fold Cross Validation for robust evaluation

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM

## 📊 Machine Learning Pipeline

1. Data Loading
2. Data Preprocessing
3. Feature Engineering
4. Feature Encoding
5. Model Training
6. Ensemble Stacking
7. Prediction Generation
8. Submission File Creation

## 🔍 Feature Engineering

- Hour, Minute, Day of Week, Month, Quarter
- Weekend and Peak Hour Indicators
- Cyclical Time Encoding (Sin/Cos)
- Geohash Prefix Features
- Road and Traffic Interaction Features
- Target Encoding for Geographical Regions

## 🤖 Models Used

- Gradient Boosting Regressor
- Random Forest Regressor
- Extra Trees Regressor
- XGBoost Regressor
- LightGBM Regressor
- Ridge Regression (Meta Learner)

## 📈 Validation Strategy

- 5-Fold Cross Validation
- Stacked Ensemble Learning
- Ridge Regression as Meta Learner

## 📂 Project Structure

```
├── train.csv
├── test.csv
├── traffic_demand_v2
├── submission.csv
└── README.md
```

## ▶️ Installation

```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```

## ▶️ Run

```bash
python traffic_demand_v2
```

The script automatically:
- Loads the dataset
- Performs feature engineering
- Trains all models
- Creates stacked predictions
- Generates `submission.csv`

## 📌 Highlights

- Engineered 30+ spatial and temporal features
- Implemented stacked ensemble learning using XGBoost, LightGBM, Random Forest, Extra Trees, and Gradient Boosting
- Applied target encoding, robust preprocessing, and cross-validation to improve prediction accuracy

## 👩‍💻 Author

**alisha waghmare**


