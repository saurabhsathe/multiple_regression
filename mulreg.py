# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 11:41:47 2018

@author: therock
"""

import matplotlib as mp
import pandas as pd
import numpy as np

dataset  = pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
le=LabelEncoder()
X[:,3]=le.fit_transform(X[:,3])
oh=OneHotEncoder(categorical_features=[3])
X=oh.fit_transform(X).toarray()

X=X[:,1:]

from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regres=LinearRegression()
regres.fit(xtrain,ytrain)

ypred=regres.predict(xtest)

import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
xopt=X[:,[0,1,2,3,4,5]]
regres_ols=sm.OLS(endog=Y,exog=xopt).fit()
print(regres_ols.summary())

xopt=X[:,[0,1,3,4,5]]
regres_ols=sm.OLS(endog=Y,exog=xopt).fit()
print(regres_ols.summary())

xopt=X[:,[0,3,4,5]]
regres_ols=sm.OLS(endog=Y,exog=xopt).fit()
print(regres_ols.summary())


xopt=X[:,[0,3,5]]
regres_ols=sm.OLS(endog=Y,exog=xopt).fit()
print(regres_ols.summary())

xopt=X[:,[0,3]]
regres_ols=sm.OLS(endog=Y,exog=xopt).fit()
print(regres_ols.summary())

#model with allin
ypred=regres.predict(xtest)

#model with optimized columns
regres2=LinearRegression()
regres2.fit(xopt[0:40,:],ytrain)
ypred2=regres2.predict(xtest)






