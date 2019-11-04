import os

from flask import Flask
from flask_api import FlaskAPI
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort
from model import *
from wtforms import Form, BooleanField, StringField, IntegerField, PasswordField, validators
from flask_httpauth import HTTPBasicAuth
import boto3
from botocore.errorfactory import ClientError

app = FlaskAPI(__name__, instance_relative_config=True)
app.static_folder = 'static'

auth = HTTPBasicAuth()

users = {
    "test": "pw"
}

class Parametersval(Form):
    GrLivArea = IntegerField('GrLivArea', validators=[validators.NumberRange(min=300, max=6000)])
    LotArea = IntegerField('LotArea', validators=[validators.NumberRange(min=1300, max=250000)])
    TotalBsmtSF = IntegerField('TotalBsmtSF', validators=[validators.NumberRange(min=0, max=6200)])
    GarageArea = IntegerField('GarageArea', validators=[validators.NumberRange(min=0, max=1500)])
    BsmtFinSF1 = IntegerField('BsmtFinSF1', validators=[validators.NumberRange(min=0, max=6000)])
    LotFrontage = IntegerField('LotFrontage', validators=[validators.NumberRange(min=20, max=350)])
    OverallCond = IntegerField('OverallCond', validators=[validators.NumberRange(min=1, max=10)])
    OverallQual = IntegerField('OverallQual', validators=[validators.NumberRange(min=1, max=10)])
    YearRemodAdd = IntegerField('YearRemodAdd', validators=[validators.NumberRange(min=1950, max=2019)])
    YearBuilt = IntegerField('YearBuilt', validators=[validators.NumberRange(min=1890, max=2019)])

#boto_sts=boto3.client('sts')
#stsresponse = boto_sts.assume_role(RoleArn="arn:aws:iam::753835687830:role/MyIam", RoleSessionName='newSess')
#newsession_id = stsresponse["Credentials"]["AccessKeyId"]
#newsession_key = stsresponse["Credentials"]["SecretAccessKey"]

@auth.get_password
def get_pw(username):
    if username in users:
        return users.get(username)
    return None

@app.route('/', methods=["GET"])
@auth.login_required
def home():   
    return {"info": "JSON API - House price prediction"}

@app.route('/predict', methods=["GET"])
@auth.login_required
def predict():    
    args = request.args
    form = Parametersval(args)
    
    client = boto3.client('s3')
    s3_key = 'model_joblib'
    bucket = 'houseprediction'

    try:
        client.head_object(Bucket=bucket,Key=s3_key)
    except ClientError:
        trainandsave()
    pass

    print(form.data['YearRemodAdd'])

    if (form.data['YearRemodAdd'] is not None) and (form.data['YearRemodAdd'] < form.data['YearBuilt']):
        return {"inputerror": "YearRemodAdd greater than YearBuilt"}, status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE


    if form.validate() == False:
        return form.errors, status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE
    else:
        pred = prepandmodelv2(form)

    return pred
