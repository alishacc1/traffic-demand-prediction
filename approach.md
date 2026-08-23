Traffic Demand Prediction – Solution Approach

Introduction



This project aims at predicting traffic demand through a combination of historical location data, roads, weather and timestamp. Given the metric of evaluation based on R² score, the approach taken will focus on developing an appropriate set of features using historical data and leveraging several machine learning algorithms with the help of stacking technique.



Data Preprocessing



The training and testing dataset were read using Pandas. For missing values handling, -999 was used to fill in after feature engineering for compatibility among all machine learning algorithms.



For encoding categorical variables, Label Encoder was used. For avoiding discrepancies between training and testing data, fitting of label encoder was done considering values from both datasets together.



Encoded features are:

* geohash
* RoadType
* LargeVehicles
* Landmarks
* Temperature
* Weather
* geo\_prefix2
* geo\_prefix3
* geo\_prefix4
* geo\_prefix5



Feature Engineering:

Feature Engineering proved to be a critical step in the solution process.

Features derived from Timestamps:

The timestamp column was converted into a date format, and following features were extracted:

* Hour
* Minute
* DayOfWeek
* Month
* Quarter
* Weekend Indicator



Cyclic Time Features:

As time is cyclic in nature, the following transformation was done for hour, minute and dow (dayofweek):

* hour\_sin
* hour\_cos
* min\_sin
* min\_cos
* dow\_sin
* dow\_cos
* month\_sin
* month\_cos



This helps models understand that 23:00 and 00:00 are close to each other.



Time-Based Behavior Features



Further Traffic Indicators Were Generated:

* time\_buckets
* morning\_peak
* evening\_peak
* night
* lunch\_hour
* rush\_weekday



They reflect realistic traffic behavior.



Geohash Features:

The geohash feature was broken down into:

* geo\_len
* geo\_prefix2
* geo\_prefix3
* geo\_prefix4
* geo\_prefix5



This makes it possible for the model to learn location-specific traffic behavior at various levels of spatial resolution.



Day Features:

Additional time indicators:

* day\_mod7
* day\_mod30
* day\_sin
* day\_cos



Interaction Features:

Additional interaction features, including:



* hour \* dayofweek
* numberoflanes \* hour



They reflect relations between different variables.



Target Encoding:

In order to learn location-based traffic behavior, geohash features underwent target encoding.



Additional Features:

* geo\_mean\_demand
* geo\_std\_demand



These represent the average and standard deviation of the historical traffic demand in each geographic area.



Further:

hour\_mean\_demand



was added using the average traffic demand for each hour of the day.



Model Development:

In order to make use of several machine learning algorithms, multiple regression models were developed.



Models Used:

* Gradient Boosting Regressor
* Extra Trees Regressor
* Random Forest Regressor
* XGBoost Regressor
* LightGBM Regressor



A combination of models increases robustness and reflects a wide range of traffic patterns.



Cross Validation:

For training purposes, a 5-Fold Cross Validation method was used.



Advantages of CV:



* Prevent overfitting
* Yield more consistent validation scores
* Provide out-of-fold predictions for stacking



Random shuffling was performed using:

random\_state = 42



Stacking Ensemble Model:

To improve results, a stacking approach was chosen.



Level 1 Models:

* Gradient Boosting
* Extra Trees
* Random Forest
* XGBoost
* LightGBM



Generated:

* OOF Predictions
* Test Predictions



Meta-Learner:

A model using ridge regression as a meta-learner.



Prior to fitting the meta-learner:

RobustScaler scaling was performed on model predictions.

The Ridge learner learned the optimal weight for each base learner predictions.

In general, stacking yields better results compared to averaging since the meta-learner learns which models work well.



Final prediction:

Demand predictions were made using the stacked ridge model.

Any negative predictions were set to 0 as traffic demands can only be positive.



Contents of submission file:

* Index
* Demand



Exported as:

submission.csv



Tools \& Libraries Used:

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* LightGBM



Key scikit-learn components used:

* RandomForestRegressor
* ExtraTreesRegressor
* GradientBoostingRegressor
* Ridge Regression
* KFold Cross Validation
* LabelEncoder
* RobustScaler



Key techniques applied:

* Feature Engineering
* Target Encoding
* Geospatial Feature Extraction
* Time-series like Features
* Ensemble Models
* Stacking using cross validation
* Ridge meta-learning
* Model blending



Expected results:

Combining feature engineering, ensembles of tree models, and stacking will allow to find both time and spatial patterns of traffic demand.

