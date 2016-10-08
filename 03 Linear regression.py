#loading CSV data with PD
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/bikeshare.csv'
bikes = pd.read_csv(url, index_col='datetime', parse_dates=True)
Correlation and head map
import seaborn as sns
bikes.corr()
sns.heatmap(bikes.corr())
Linear regression with one variable 
#create X and y
feature_cols = ['temp']
X=bikes[feature_cols]
y =bikes.total

#instantiate and fit
from sklearn.linear_model import LinearRegression 
linreg = LinearRegression()
linreg.fit(X, y)

#print the coefficients
print linreg.intercept_
print linreg.coef_

# pair the feature names with the coefficients
zip(feature_cols, linreg.coef_)
#linear regression normal with multiple variables 
# create X and y
feature_cols = ['temp', 'season', 'weather', 'humidity']
X = bikes[feature_cols]
y = bikes.total

# instantiate and fit
linreg = LinearRegression()
linreg.fit(X, y)

# print the coefficients
print linreg.intercept_
print linreg.coef_

# pair the feature names with the coefficients
zip(feature_cols, linreg.coef_)

#splitting training set.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
# Regularization with  RidgeCV (different reg parmsalphas)
Regularized cost function as mentioned below.

>>> from sklearn import linear_model
>>> clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
>>> clf.fit(X_train, y_train)       
RidgeCV(alphas=[0.1, 1.0, 10.0], cv=None, fit_intercept=True, scoring=None,
    normalize=False)
pred=clf.predict(X_test)
>>> clf.alpha_                                      
0.1
#regularization with  LassoCV(different reg parmsalphas)
Regularized cost function as mentioned below.

>>> from sklearn import linear_model
>>> clf = linear_model.LassoCV(alphas=[0.1, 1.0, 10.0])
>>> clf.fit(X_train, y_train)       
Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
>>>pred= clf.predict(X_test)
array([ 0.8])
# calculate metrics!

from sklearn import metrics
import numpy as np
print 'MAE:', metrics.mean_absolute_error(true, pred)
print 'MSE:', metrics.mean_squared_error(true, pred)
print 'RMSE:', np.sqrt(metrics.mean_squared_error(true, pred))

RMSE:155.649459131---With Ridge
RMSE:155.643749947---With Losso
One variable temp
RMSE:166.175955908---With normal
RMSE:166.175913119---With RidgeCV
RMSE:166.175581741---With LASSO


#Normalize the features 
LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)

LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True,normalize=False, precompute='auto', max_iter=1000, tol=0.0001, copy_X=True, cv=None, verbose=False, n_jobs=1,positive=False, random_state=None, selection='cyclic')
RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, scoring=None,cv=None, gcv_mode=None, store_cv_values=False)

normalize : boolean, optional, default False
If True, the regressors X will be normalized before regression.

fit_intercept : boolean
Whether to calculate the intercept for this model. If set to false, no intercept will be used in calculations (e.g. data is expected to be already centered)

alphas : numpy array, optional
List of alphas where to compute the models. If None alphas are set automatically
#difference between Ridge and LASSO
Ridge and Lasso regression uses two different penalty functions. Ridge uses l2 where as lasso go with l1. In ridge regression, the penalty is the sum of the squares of the coefficients and for the Lasso, it's the sum of the absolute values of the coefficients.
"""