#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split


# load data_set
iris = datasets.load_iris()

# extract x & y of 2 classes only
features = iris.data[:100, :]
target = iris.target[:100]

# split into train and test data
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# classification for train data
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
transformed_x_train = lda.transform(x_train)
transformed_x_test = lda.transform(x_test)

# print coefficients
print """
coefficients  : %s
feature_names : %s
intercept     : %s
""" % (lda.coef_, iris.feature_names, lda.intercept_)

# plot training data
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(20, 8))
axL.set_title("Training", fontsize=40)
axL.hist(transformed_x_train[np.where(transformed_x_train < 0)], bins=10, normed=True, color='red')
axL.hist(transformed_x_train[np.where(transformed_x_train > 0)], bins=10, normed=True, color='blue')

# plot test data
axR.set_title("Test", fontsize=40)
axR.hist(transformed_x_test[np.where(transformed_x_test < 0)], bins=5, normed=True, color='red')
axR.hist(transformed_x_test[np.where(transformed_x_test > 0)], bins=5, normed=True, color='blue')
fig.show()

# get accuracy
accuracy = lda.score(x_test, y_test)
print "accuracy is %f" % accuracy
