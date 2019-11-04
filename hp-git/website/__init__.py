from flask import Flask
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from wtforms import Form, BooleanField, StringField, IntegerField, PasswordField, validators
import urllib

import requests

app = Flask(__name__)

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

@app.route('/')
def form_template():
    return render_template('home.html')

@app.route('/getpredictions')
def get_data():
    args = request.args
    form = Parametersval(args)
    url_params = urllib.parse.urlencode(form.data)
    #url = "http://127.0.0.1:8001/predict?"
    #url = "http://web:8080/predict?"
    url = "http://3.133.132.231:8080/predict?"
    final_url = url + url_params
    print(final_url, flush=True)

    try:
        r = requests.get(final_url, auth=('test', 'pw'))
    except requests.ConnectionError as e:
        print(e, flush=True)
        return "ERRO2"

    return r.json()