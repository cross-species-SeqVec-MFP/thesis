import numpy as np
import pickle
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
import sys

modelPath = sys.argv[1]
lengthPath = sys.argv[2]
testsetFile = sys.argv[3]

with open(lengthPath, 'rb') as f:
    len_pro = pickle.load(f)




with open(modelPath + '/performance.pkl', 'rb') as f:
    dd = pickle.load(f)

f1 = dd['pc']


with open(testsetFile, 'rb') as fw:
    _, Ytest, _ = pickle.load(fw)



num_annotations = np.sum(Ytest, axis=0)
num_annotations_protein = np.sum(Ytest, axis=1)


from scipy import stats
print(stats.spearmanr(len_pro, f1))
print(stats.spearmanr(f1, num_annotations_protein))
