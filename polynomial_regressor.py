# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 19:24:16 2018

@author: RAJ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X,y)
y_pred = linear_regressor.predict(6.5)

from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree = 4)
X_poly = polynomial_regressor.fit_transform(X)
linear_regressor_poly = LinearRegression()
linear_regressor_poly.fit(X_poly,y)
y_pred_poly = linear_regressor_poly.predict(polynomial_regressor.fit_transform(6.5))

