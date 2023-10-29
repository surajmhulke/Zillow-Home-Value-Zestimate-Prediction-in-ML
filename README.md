# Zillow-Home-Value-Zestimate-Prediction-in-ML

In this article, we will try to implement a house price index calculator which revolutionized the whole real estate industry in the US. This will be a regression task in which we have been provided with logarithm differences between the actual and the predicted prices of those homes by using a benchmark model.

# Importing Libraries and Dataset
Python libraries make it easy for us to handle the data and perform typical and complex tasks with a single line of code.

Pandas – This library helps to load the data frame in a 2D array format and has multiple functions to perform analysis tasks in one go.
Numpy – Numpy arrays are very fast and can perform large computations in a very short time.
Matplotlib/Seaborn – This library is used to draw visualizations.
Sklearn – This module contains multiple libraries are having pre-implemented functions to perform tasks from data preprocessing to model development and evaluation.
XGBoost – This contains the eXtreme Gradient Boosting machine learning algorithm which is one of the algorithms which helps us to achieve high accuracy on predictions.
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn import metrics 
from sklearn.svm import SVC 
from xgboost import XGBRegressor 
from sklearn.linear_model import LinearRegression, Lasso, Ridge 
from sklearn.ensemble import RandomForestRegressor 
  
import warnings 
warnings.filterwarnings('ignore') 
Now let’s load the dataset into the panda’s data frame and print its first five rows.

df = pd.read_csv('Zillow.csv') 
df.head()
Output:


First five rows of the dataset.
 

Now let’s check the size of the dataset.

df.shape
Output:

(90275, 59)
The dataset contains too many features but we can observe that there are null values in the dataset as well. So, before performing any of the analyses let’s first clean the data.

# Data Cleaning
The data which is obtained from the primary sources is termed the raw data and required a lot of preprocessing before we can derive any conclusions from it or do some modeling on it. Those preprocessing steps are known as data cleaning and it includes, outliers removal, null value imputation, and removing discrepancies of any sort in the data inputs.

to_remove = [] 
for col in df.columns: 
  
    # Removing columns having only one value. 
    if df[col].nunique() == 1: 
        to_remove.append(col) 
  
    # Removing columns with more than 90% of the 
    # rows as null values. 
    elif (df[col].isnull()).mean() > 0.60: 
        to_remove.append(col) 
  
print(len(to_remove)) 
Output:

30
So, in total there are 30 columns that contain either unique values equal to 1 or have around 60% of the rows as null values.

df.drop(to_remove, 
        axis=1, 
        inplace=True) 
Let’s check which column of the dataset contains which type of data.

df.info()
Output:

# Information regarding data in the columns
 
Here we can see that there are still null values in different columns of the dataset. So, let’s check for the null values in the data frame and impute them by using the mean value for the continuous variables and the mode value for the categorical columns.

df.isnull().sum().plot.bar() 
plt.show()
Output:

# Bar plot for the count of the number of null values in the columns
 
for col in df.columns: 
    if df[col].dtype == 'object': 
        df[col] = df[col].fillna(df[col].mode()[0]) 
    elif df[col].dtype == np.number: 
        df[col] = df[col].fillna(df[col].mean()) 
  
df.isnull().sum().sum() 
Output:

0

# Exploratory Data Analysis
EDA is an approach to analyzing the data using visual techniques. It is used to discover trends, and patterns, or to check assumptions with the help of statistical summaries and graphical representations. 

ints, objects, floats = [], [], [] 
  
for col in df.columns: 
    if df[col].dtype == float: 
        floats.append(col) 
    elif df[col].dtype == int: 
        ints.append(col) 
    else: 
        objects.append(col) 
  
len(ints), len(floats), len(objects) 
Output:

(1, 26, 2)
for col in objects: 
    print(col, ' -> ', df[col].nunique()) 
    print(df[col].unique()) 
    print() 
Output:

Unique value count as well as the unique values in the categorical columns
 

# The number of unique values is too high to visualize else we could have plotted a count plot for these categorical columns.

plt.figure(figsize=(8, 5)) 
sb.distplot(df['target']) 
plt.show() 
Output:

Distribution plot for the target column
![image](https://github.com/surajmhulke/Zillow-Home-Value-Zestimate-Prediction-in-ML/assets/136318267/131816b6-c65e-4eaf-af73-caf2abb80bc1)


# From the above distribution plot of the target variable, it seems like there are outliers in the data. Let’s use a boxplot to detect them.

plt.figure(figsize=(8, 5)) 
sb.boxplot(df['target']) 
plt.show() 
Output:

Box plot to detect outliers in the target column
![image](https://github.com/surajmhulke/Zillow-Home-Value-Zestimate-Prediction-in-ML/assets/136318267/2f199309-5e33-4efb-96c9-95987b56f86e)


# From the above box plot, we can clip the target values between -1 to 1 for feasible(only on patterns) training of the model.

print('Shape of the dataframe before removal of outliers', df.shape) 
df = df[(df['target'] > -1) & (df['target'] < 1)] 
print('Shape of the dataframe after removal of outliers ', df.shape) 
Output:

"""print('Shape of the dataframe before removal of outliers', df.shape) 
df = df[(df['price'] > -1) & (df['price'] < 1)] 
print('Shape of the dataframe after removal of outliers ', df.shape) """
#outliers where removing entire datasets so prefer to go by IQR method to remove outliers
This means that we  lose that all points.

for col in objects: 
    le = LabelEncoder() 
    df[col] = le.fit_transform(df[col]) 
Now let’s check whether there are any highly correlated features in our dataset or not. 

plt.figure(figsize=(15, 15)) 
sb.heatmap(df.corr() > 0.8, 
           annot=True, 
           cbar=False) 
plt.show() 
Output:

Heat map to find highly correlated features.
![image](https://github.com/surajmhulke/Zillow-Home-Value-Zestimate-Prediction-in-ML/assets/136318267/670e7f0c-6799-4dad-8b2d-e15539ca32ef)


Certainly, there are some highly correlated features in the feature space. We will remove them to decrease the complexity of the data and avoid any cases of error.

to_remove = ['calculatedbathnbr', 'fullbathcnt', 'fips', 
             'rawcensustractandblock', 'taxvaluedollarcnt', 
             'finishedsquarefeet12', 'landtaxvaluedollarcnt'] 
  
df.drop(to_remove, axis=1, inplace=True) 
Model Training
Now we will separate the features and target variables and split them into training and testing data by using which we will select the model which is performing best on the validation data.

features = df.drop(['parcelid'], axis=1) 
target = df['target'].values 
  
X_train, X_val,\ 
    Y_train, Y_val = train_test_split(features, target, 
                                      test_size=0.1, 
                                      random_state=22) 
X_train.shape, X_val.shape 
Output:

((720, 12), (80, 12))

# Normalizing the data before feeding it into machine learning models helps us to achieve stable and fast training.

Normalizing the features for stable and fast training. 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_val = scaler.transform(X_val) 
We have split our data into training and validation data also the normalization of the data has been done. Now let’s train some state-of-the-art machine learning models and select the best out of them using the validation dataset.

from sklearn.metrics import mean_absolute_error as mae 
models = [LinearRegression(), XGBRegressor(), 
          Lasso(), RandomForestRegressor(), Ridge()] 
  
for i in range(5): 
    models[i].fit(X_train, Y_train) 
  
    print(f'{models[i]} : ') 
  
    train_preds = models[i].predict(X_train) 
    print('Training Error : ', mae(Y_train, train_preds)) 
  
    val_preds = models[i].predict(X_val) 
    print('Validation Error : ', mae(Y_val, val_preds)) 
    print() 
Output:
LinearRegression() : 
Training Error :  504373.5411832306
Validation Error :  399103.62977603

XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=None, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=None, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=None, n_jobs=None,
             num_parallel_tree=None, random_state=None, ...) : 
Training Error :  8811.670128038195
Validation Error :  201277.8509033203

Lasso() : 
Training Error :  504372.84216277895
Validation Error :  399102.74087378226

RandomForestRegressor() : 
Training Error :  122632.76930555556
Validation Error :  189605.45512499998

Ridge() : 
Training Error :  503799.7421691335
Validation Error :  398476.37660499534

