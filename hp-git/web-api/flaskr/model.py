import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
from sklearn.impute import SimpleImputer
from eli5.sklearn import PermutationImportance
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn import metrics
from xgboost import XGBRegressor
from wtforms import Form, BooleanField, StringField, IntegerField, PasswordField, validators
from flask_api import status
import boto3
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.externals import joblib

mj = None
rmse = None

#boto_sts=boto3.client('sts')
#stsresponse = boto_sts.assume_role(RoleArn="arn:aws:iam::753835687830:role/MyIam", RoleSessionName='new')
#newsession_id = stsresponse["Credentials"]["AccessKeyId"]
#newsession_key = stsresponse["Credentials"]["SecretAccessKey"]
s3 = boto3.client('s3')

def prepandmodelv2(form):
    global mj
    global rmse

    selected = list(form.data.keys())
    data = {}

    if (mj is None):
        response = s3.get_object(Bucket='houseprediction', Key='model_joblib')
        body = response['Body'].read()
        mj = pickle.loads(body)
        #mj = joblib.load('/home/ubuntu/house-prediction/model_joblib')

    if (rmse is None):
        response = s3.get_object(Bucket='houseprediction', Key='error_joblib')
        body = response['Body'].read()
        rmse = pickle.loads(body)
        #rmse = joblib.load('/home/ubuntu/house-prediction/error_joblib')

    for i in range(len(selected)):
        a=list(form.data.values())[i]
        data.update( {str(i) : a})

    df = pd.DataFrame(data, index=[0])

    preds = mj.predict(df)
    preds = np.exp(preds)
    
    return {"prediction": float(preds[0]), "rmse":float(rmse)}

def trainandsave():
    X_full = pd.read_csv('/home/ubuntu/hp/hp-git/web-api/input/train.csv')
    X_test_full = pd.read_csv('/home/ubuntu/hp/hp-git/web-api/input/test.csv')
    print('Train shape:', X_full.shape)
    print('Test shape:', X_test_full.shape)

    selected = ['GrLivArea', 'LotArea', 'TotalBsmtSF', 'GarageArea', 'BsmtFinSF1', 'LotFrontage', 'YearBuilt', 'YearRemodAdd', 'OverallCond', 'OverallQual']

    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = np.log(X_full.SalePrice)
    X_full.drop(['SalePrice'], axis=1, inplace=True)

    ind = X_test_full.Id

    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.7, random_state=0)

    X_train = X_train_full[selected].copy()
    X_valid = X_valid_full[selected].copy()
    X_test = X_test_full[selected].copy()
    X_full = X_full[selected].copy()

    print('TEST: ', X_test.shape)
    print('TRAIN: ', X_valid.shape)

    object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
    good_label_cols = [col for col in object_cols if set(X_train[col]) == set(X_valid[col])]
    bad_label_cols = list(set(object_cols)-set(good_label_cols))
    X_train = X_train.drop(bad_label_cols, axis=1)
    X_valid = X_valid.drop(bad_label_cols, axis=1)
    X_test = X_test.drop(bad_label_cols, axis=1)
    X_full = X_full.drop(bad_label_cols, axis=1)

    label_encoder = LabelEncoder()

    for col in good_label_cols:
        X_train[col] = label_encoder.fit_transform(X_train[col])
        X_valid[col] = label_encoder.transform(X_valid[col])
        X_test[col] = label_encoder.fit_transform(X_test[col])
        X_full[col] = label_encoder.fit_transform(X_full[col])

    my_imputer = SimpleImputer() 
    X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    X_valid = pd.DataFrame(my_imputer.transform(X_valid))
    X_test = pd.DataFrame(my_imputer.fit_transform(X_test))
    X_full = pd.DataFrame(my_imputer.fit_transform(X_full))

    X_train.columns = X_train.columns
    X_valid.columns = X_valid.columns
    X_test.columns = X_test.columns
    X_full.columns = X_full.columns

    eval_set = [(X_valid, y_valid)]

    model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, random_state=42) 
    model.fit(X_train, y_train, eval_metric="rmse", eval_set=eval_set, early_stopping_rounds=20, verbose=True)

    predictions_error = model.predict(X_valid)
    rmse = np.sqrt(metrics.mean_squared_error(predictions_error, y_valid))

    print(rmse)

    #joblib.dump(model, 'model_joblib')
    #joblib.dump(rmse, 'error_joblib')
    
    #boto_sts=boto3.client('sts')
    #stsresponse = boto_sts.assume_role(RoleArn="arn:aws:iam::753835687830:role/MyIam", RoleSessionName='new')
    #newsession_id = stsresponse["Credentials"]["AccessKeyId"]
    #newsession_key = stsresponse["Credentials"]["SecretAccessKey"]

    bucket='houseprediction'
    key='model_joblib'
    pickle_byte_obj = pickle.dumps(model)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket,key).put(Body=pickle_byte_obj)
 
    key_error='error_joblib'
    pickle_byte_obj_error = pickle.dumps(rmse)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket,key_error).put(Body=pickle_byte_obj_error)
