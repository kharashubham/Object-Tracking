from sklearn.ensemble import GradientBoostingClassifier
from feature_extraction import get_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import metrics as slm
from sklearn.model_selection import train_test_split


score = 0
def classify():

    iterators = [100, 200, 300, 400, 500]
    GBTrErr = []
    GBVaErr = []
    y, x = get_dataset()

    Xtr, Xva, Ytr, Yva = train_test_split(x, y, test_size=0.20, random_state=42)
    
    for j, i in enumerate(iterators):
        est = GradientBoostingClassifier(n_estimators=i, learning_rate=0.1, max_depth=5, random_state=0, loss='ls')
        est.fit(Xtr, Ytr)
        score = est.score(x, y)
        if j == len(iterators) - 1:
            GBYtrpred = est.predict(Xtr)
            GBYvapred = est.predict(Xva)
            GBTrErr.append(mean_squared_error(Ytr, GBYtrpred))
            GBVaErr.append(mean_squared_error(Yva, GBYvapred))
        else:
            GBTrErr.append(mean_squared_error(Ytr, est.predict(Xtr)))
            GBVaErr.append(mean_squared_error(Yva, est.predict(Xva)))

    plt.plot(iterators, GBTrErr)
    plt.plot(iterators, GBVaErr)
    plt.show()

def get_score():
    return score



