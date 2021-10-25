import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
import sys

Cvalue = [1e-3, 1e-1, 10, 1e-2, 1, 100]

trainingsetFile = sys.argv[1]
modelPath = sys.argv[2]

#this data is scaled already
with open(trainingsetFile, 'rb') as fw:
    X_train, Y_train, GO_terms = pickle.load(fw)

for C in Cvalue:
    clf = MultiOutputClassifier(SGDClassifier(loss='log', penalty='l2', alpha=C, max_iter=1000, random_state=0))
    clf.fit(X_train, Y_train)

    with open(modelPath + '/c%s_LR.pkl' % C, 'wb') as fw:
        pickle.dump({'trained_model': clf}, fw, protocol=4)
