# Import Libraries
import numpy as np
import pandas as pd
import os, missingno, joblib
from datasist.structdata import detect_outliers
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder ,OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.metrics import mean_squared_error,confusion_matrix,classification_report

# Reading Dataset
File_Path = os.path.join(os.getcwd(),'avocado.csv')
df = pd.read_csv(File_Path)

df.columns = df.columns.str.replace(' ', '_')
df.columns=df.columns.str.lower()

# Renaming columns
df.rename(columns={'4046': 'plus_4046', '4225': 'plus_4225', '4770': 'plus_4770'}, inplace=True)

df.drop(['unnamed:_0','date'], axis = 1, inplace = True)
X =df.drop(columns=['averageprice'], axis=1)
y = df['averageprice']


# Feature Selection
# Feature Selection is a techinque of finding out the features that contribute the most to our model i.e. the best predictors.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, shuffle=True, random_state=45)
num_cols = df.select_dtypes(include='number').columns.to_list()
categ_cols = df.select_dtypes(include='object').columns.tolist()
num_cols1 = df.select_dtypes(include='number').columns.tolist()[1:]

# Pipeline
# Create separate pipelines for numeric and categorical columns
# For Numeric
num_pipeline = Pipeline(steps=[
                                ('selector', DataFrameSelector(num_cols1)),
                                ('scaler', StandardScaler())
])

# For Categorical
categ_pipeline = Pipeline(steps=[
                                ('selector', DataFrameSelector(categ_cols)),
                                ('OHe',OneHotEncoder())
                                ])


## all pipline
all_pipline = FeatureUnion(transformer_list=[
                ('numerical', num_pipeline),
                ('categorical', categ_pipeline)

])

_ = all_pipline.fit(X_train)

def process_new(X_new):
    df_new = pd.DataFrame([X_new])
    df_new.columns = X_train.columns

    # Adjust the Datatypes
    df_new['total_volume'] = df_new['total_volume'].astype('float')
    df_new['plus_4046'] = df_new['plus_4046'].astype('float')
    df_new['plus_4225'] = df_new['plus_4225'].astype('float')
    df_new['plus_4770'] = df_new['plus_4770'].astype('float')
    df_new['total_bags'] = df_new['total_bags'].astype('float')
    df_new['small_bags'] = df_new['small_bags'].astype('float')
    df_new['large_bags'] = df_new['large_bags'].astype('float')
    df_new['xlarge_bags'] = df_new['xlarge_bags'].astype('float')
    df_new['type'] = df_new['type'].astype('str')
    df_new['year'] = df_new['year'].astype('int')
    df_new['region'] = df_new['region'].astype('str')
  
    X_processed = all_pipline.transform(df_new)
    
    return X_processed

