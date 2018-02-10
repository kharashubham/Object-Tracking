from sklearn.ensemble import GradientBoostingClassifier
#from feature_extraction import get_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import metrics as slm
from sklearn.model_selection import train_test_split


score = 0
def classify():

    iterators = [100, 200, 300, 400, 500]
    GBTrErr = []
    GBVaErr = []
    #y, x = get_dataset()
    data = pd.read_csv('dataset.csv')
    print(data.head())
    y = data.target 
    X = data.drop('target', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

    for j, i in enumerate(iterators):
        est = GradientBoostingClassifier(n_estimators=i, learning_rate=0.1, max_depth=5, random_state=0)
        est.fit(X_train, y_train)
        #score = est.score(X, y)
        if j == len(iterators) - 1:
            GBYtrpred = est.predict(X_train)
            GBYvapred = est.predict(X_test)
            GBTrErr.append(mean_squared_error(y_train, GBYtrpred))
            GBVaErr.append(mean_squared_error(y_test, GBYvapred))
        else:
            GBTrErr.append(mean_squared_error(y_train, est.predict(X_train)))
            GBVaErr.append(mean_squared_error(y_test, est.predict(X_test)))

    #plt.plot(iterators, GBTrErr)
    plt.plot(iterators, GBVaErr)
    plt.show()

def get_score():
    return score

classify()

