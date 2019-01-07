#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# load data_set
boston = datasets.load_boston()

# split into train and test data
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

# regression for train data
linear_regression_model = LinearRegression()
linear_regression_model.fit(x_train, y_train)

# print coefficients
print """
coefficients  : %s
feature_names : %s
intercept     : %s
""" % (linear_regression_model.coef_, boston.feature_names, linear_regression_model.intercept_)

# get average  error
average_error = np.mean(abs(linear_regression_model.predict(x_test) - y_test))
print "average error is %f" % average_error
