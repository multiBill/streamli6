import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator , TransformerMixin
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , precision_score , recall_score
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

df = pd.read_csv("train.csv")
x = df.drop('Survived',axis=1)
y = df.Survived

class PrepProcesor(BaseEstimator, TransformerMixin): 
    def fit(self, x, y=None): 
        self.ageImputer = SimpleImputer()
        self.ageImputer.fit(x[['Age']])        
        return self 
        
    def transform(self, x, y=None):
        x['Age'] = self.ageImputer.transform(x[['Age']])
        x['CabinClass'] = x['Cabin'].fillna('M').apply(lambda x: str(x).replace(" ", "")).apply(lambda x: re.sub(r'[^a-zA-Z]', '', x))
        x['CabinNumber'] = x['Cabin'].fillna('M').apply(lambda x: str(x).replace(" ", "")).apply(lambda x: re.sub(r'[^0-9]', '', x)).replace('', 0) 
        x['Embarked'] = x['Embarked'].fillna('M')
        x = x.drop(['PassengerId', 'Name', 'Ticket','Cabin'], axis=1)
        return x
    
numeric_pipeline = Pipeline([('scaler',StandardScaler())])
categorical_pipeline = Pipeline([('onehot',OneHotEncoder(handle_unknown='ignore'))])
transformer = ColumnTransformer([( 'num',numeric_pipeline,['Pclass','Age','SibSp','Parch','Fare','CabinNumber'] ),('cat',categorical_pipeline,['Sex','CabinClass','Embarked'])])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.10, random_state=1234)
mlpipe = Pipeline([('preprocessor',PrepProcesor() ),( 'Transformer' , transformer),('xgb',XGBClassifier() )])
mlpipe.fit(x_train,y_train)
y_hat = mlpipe.predict(x_test)   
columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked']

st.title('Will you survive if you were among Titanic passengers or not :ship:')
passengerid = st.text_input("Input Passenger ID", '8585') 
pclass = st.selectbox("CHoise Class " , [1,2,3])
name = st.text_input("input passenger name : ","mhmmd jalal davooodian")
sex = st.radio('CHoise sex' , ['male','female'])
age = st.slider("choise age ",0,1000)
sibsp = st.slider("choise sibsp ",0,20)
parch = st.slider("Choose parch",0,10)
ticket = st.text_input('input ticket number :',8585)
fare = st.number_input("input fare price :",0,1000)
cabin = st.text_input("input cabin",'C52')
embarked = st.selectbox('select',['S','D','Q'])
def predict():
    row = np.array([passengerid,pclass,name,sex,age,sibsp,parch ,ticket,fare,cabin,embarked])
    x = pd.DataFrame([row],columns=columns )
    prediction = mlpipe.predict(x)
    if prediction[0] == 1: 
        st.success('Passenger Survived :thumbsup:')
    else: 
        st.error('Passenger did not Survive :thumbsdown:') 
st.button('predict',on_click=predict)            