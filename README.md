# - Load prediction of partially observed (digitalized) systems using Machine Learning

Author: Inas AL-Kamachy

## Data Analysis

### Load and Read the Data
he CityLearn dataset was used which is part of the OpenAI Gym
Environment. Many researchers used this environment to implement their work
related to reinforcement learning and regression tasks, as well as to perform different
competitions related to energy management, particularly in the building sector. The
data were collected in three years from January 1, 2021, to December 30, 2023, and
the data were collected from nine different buildings including Offices, Fast_food,
Standalone_retail, Strip mall_retail, and Five Multi_family buildings.
### Cluster the buildings base on Energy consumption values:
The essential step in this thesis is to group the buildings according to their similarity
and then specify the source and target building. The main limitation of the K-mean
clustering is using Euclidean as a base metric, which is unstable to the time shift. For
this reason, hierarchical clustering combined with dynamic time warping (DTW) was
used to get the best alignment between building identification, as well as to minimize
the distortion effect in time and shift
### Preprocess dataset
The data of the nine buildings were pre-processed and prepared for the time series
model. The first step was to remove the features that contained NAN values and
duplicate values. For further analysis and visualization, the date-time was set as an
index with hourly frequency, starting from 2021-01-01 to 2024-01-01.
The next step was to replace the outlier with the median of the column depending on
the Interquartile Range (IQR). After that, the data for each building were merged with
the weather and carbon-intensity data sets.
Soothing techniques were implemented to provide the ability to uncover patterns while
maintaining important features. For this purpose, Savitzky - Golay was implemented
for building-5, building-6, building-7, building-8, and building-9, where more complex
and fluctuating EEC usage patterns were observed due to the daily habits of the
clients. On the other hand, building-1,
building-2, building-3, and building-4, were more predictable and less complex. The stationary test was applied for each one of the
buildings using the ADFuller method, which was matched to all the buildings being
stationary.

### Feature Engineering
The next step was to use feature engineering to extract meaningful features using target
knowledge. It is considered as an essential process in time series analysis to get a
high performance predictive model, which can be performed by capturing temporal
dependencies and patterns. In this thesis, the lag feature was used by shifting the
energy consumption in a specific time step depending on the partial autocorrelation
pattern. This will help to capture the seasonality of the day. Sine /
cosine encoding was used to capture the cyclical patterns (hour, month). Calender
features were additional characteristics that helped the predictive model capture
trends and seasonality.
### Feature Selection
PCA was used for both the features and target as well for all of the buildings. To reduce the complexity, while mantain the important features. 
## ML Algorithms
We use four different ML algorithms:

- XGBoost
- RandomForest
- LStM

## Instance Based TL model
A native XGBoost was used to train the based model for Buildings 4, 7, and 9 instead
of XGBoostRegressor, which is considered a higher-level wrapper. XGBoost is a
powerful model for gradient boosting, which provides the ability to control everything
explicitly, particularly in fine-tuning, and the ability to incrementally train the pre-
trained models using load_model and the xgb_model methods in xgb.train. For the
implementation, the parameters of the model need to be defined in a dictionary and
the dataset needs to be converted into an optimized data structure named DMatrix, this
will enhance the memory efficiency and speed during model training and forecasting
as well.



## Results

### Visualization of RandomForestRegressor
![Best Model Visualization](best_model.png)

### Models Evaluations
# Energy Consumption (EC) Target Models Results

| Cluster         | Building      | Strategy   | MAE   | MSE    | RMSE  | RMSE %    |
|-----------------|---------------|------------|-------|--------|-------|-----------|
| **Cluster_1**   | Building_1    | No_TL      | 3.959 | 45.843 | 6.771 | -         |
|                 |               | TL_20%     | 4.528 | 50.194 | 7.085 | -4.63%    |
|                 |               | TL_40%     | 4.181 | 47.523 | 6.894 | -1.82%    |
|                 |               | TL_60%     | 4.071 | 45.012 | 6.709 | **0.062%**|
|                 | Building_2    | No_TL      | 0.038 | 0.01   | 0.101 | -         |
|                 |               | TL_20%     | 0.235 | 0.119  | 0.346 | -242%     |
|                 |               | TL_40%     | 0.172 | 0.067  | 0.259 | 156.44%   |
|                 |               | TL_60%     | 0.15  | 0.052  | 0.227 | **-124.75%** |
|                 | Building_3    | No_TL      | 0.217 | 0.235  | 0.485 | -         |
|                 |               | TL_20%     | 0.323 | 0.297  | 0.545 | -12.37%   |
|                 |               | TL_40%     | 0.255 | 0.178  | 0.422 | **12.98%**|
|                 |               | TL_60%     | 0.244 | 0.19   | 0.435 | 10.31%    |
| **Cluster_2**   | Building_5    | No_TL      | 0.705 | 0.805  | 0.897 | -         |
|                 |               | TL_20%     | 0.747 | 0.892  | 0.944 | −5.24%    |
|                 |               | TL_40%     | 0.679 | 0.736  | 0.858 | 4.35%     |
|                 |               | TL_60%     | 0.675 | 0.729  | 0.854 | **4.80%** |
|                 | Building_6    | No_TL      | 0.766 | 0.938  | 0.969 | -         |
|                 |               | TL_20%     | 0.915 | 1.31   | 1.31  | -35.18%   |
|                 |               | TL_40%     | 0.761 | 0.924  | 0.961 | 0.83%     |
|                 |               | TL_60%     | 0.742 | 0.893  | 0.945 | **2.48%** |
| **Cluster_3**   | Building_9    | No_TL      | 1.07  | 1.883  | 1.372 | -         |
|                 |               | TL_20%     | 1.25  | 2.65   | 1.628 | −18.67%   |
|                 |               | TL_40%     | 1.063 | 1.856  | 1.362 | **0.73%** |
|                 |               | TL_60%     | 1.121 | 2.046  | 1.43  | −4.23%    |



### Conclusions

The proposed method provides a significant improvement in reducing computational
resources and saving time. Additionally, it avoids developing individual models and
training from scratch for each building. Furthermore, the pre-trained model is robust
and accurate as it captures the most patterns and knowledge from a vast source domain
and transfers this knowledge to the target within a limited dataset. The proposed
method provides a meaningful framework for forecasting energy consumption (EC)
for the next 24 hours using a limited dataset.
## GitHub Repository
[GitHub Repository Link](https://github.com/InasALKamachy/EnergyConsumption_CityLearn/tree/C)
