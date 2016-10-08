#Simple logistic regression 
#load Data
import pandas as pd
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
col_names = ['id','ri','na','mg','al','si','k','ca','ba','fe','glass_type']
glass = pd.read_csv(url, names=col_names, index_col='id')
glass.sort('al', inplace=True)
glass.head()


col_names = ['id','ri','na','mg','al','si','k','ca','ba','fe','glass_type']
glass=pd.read_csv(‘glass.csv’,names=col_names,index_col='id')
glass.sort('al', inplace=True)
glass.head()

# types 1, 2, 3 are window glass
# types 5, 6, 7 are household glass
glass['household'] = glass.glass_type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})
glass.head()
#split data into train and test set and carryout LR
feature_cols = ['ri','na','mg','al','si','k','ca','ba','fe']
X = glass[feature_cols]
y = glass.household
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
#Utility Details:
X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.33, random_state=42)

train_size : float, int, or None (default is None)
If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.
random_state : int or RandomState
Pseudo-random number generator state used for random sampling.
#fitting data into logistic regression

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
feature_cols = ['ri','na','mg','al','si','k','ca','ba','fe']
X = glass[feature_cols]
y = glass.household
logreg.fit(X, y)
glass['household_pred_class'] = logreg.predict(X)


#model evaluation 
y_pred=logreg.predict(X_test)
y_pred=list(y_pred)
y_test=list(y_test)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
confusion_matrix = confusion_matrix(y_test, y_pred)
print 'Accuracy: ', accuracy_score(y_test, y_pred)
Accuracy is :0.666666666667
Accuracy is :0.96511627907(when split train:test::60:40
#intercept and coefficients 
logreg.coef_   <-type is array
logreg.intercept_

#utility Details
LogisticRegressionCV(Cs=10, fit_intercept=True, cv=None, dual=False,penalty='l2', scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=1, verbose=0,refit=True, intercept_scaling=1.0, multi_class='ovr', random_state=None)

solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’}
Algorithm to use in the optimization problem.
For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ is
faster for large ones.

Use ‘sag’  <- Explained by Andrew NJ   stochastic average gradient .

penalty : str, ‘l1’ or ‘l2’
Used to specify the norm used in the penalization. The newton-cg and lbfgs solvers support only l2 penalties.

Use ‘l2’ LASSO  regularized <- Explained by Andrew NJ

Cs : list of floats | int
Each of the values in Cs describes the inverse of regularization strength. If Cs is as an int, then a grid of Cs values are chosen in a logarithmic scale between 1e-4 and 1e4. Like in support vector machines, smaller values specify stronger regularization.

