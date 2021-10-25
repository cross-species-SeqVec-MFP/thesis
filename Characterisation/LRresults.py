import numpy as np
import pickle
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils import resample
import sys


validationsetFile = sys.argv[1]
testsetFile = sys.argv[2]
modelPath = sys.argv[3]

Cs = [1e-3, 1e-2, 1e-1, 1, 10, 100]


with open(validationsetFile, 'rb') as fw:
    Xvalid, Yvalid, GO_terms = pickle.load(fw)


def fmax(Ytrue, Ypost1):
    thresholds = np.linspace(0.0, 1.0, 51)
    ff = np.zeros(thresholds.shape, dtype=object)
    pr = np.zeros(thresholds.shape, dtype=object)
    rc = np.zeros(thresholds.shape, dtype=object)
    rc_avg = np.zeros(thresholds.shape)
    pr_avg = np.zeros(thresholds.shape)
    coverage = np.zeros(thresholds.shape)

    Ytrue = Ytrue.transpose()
    Ypost1 = Ypost1.transpose()
    tokeep = np.where(np.sum(Ytrue, 0) > 0)[0]
    Ytrue = Ytrue[:, tokeep]
    Ypost1 = Ypost1[:, tokeep]

    for i, t in enumerate(thresholds):
        _ , rc[i], _ , _ = precision_recall_fscore_support(Ytrue, (Ypost1 >= t).astype(int))
        rc_avg[i] = np.mean(rc[i])

        tokeep = np.where(np.sum((Ypost1 >= t).astype(int), 0) > 0)[0]
        Ytrue_pr = Ytrue[:, tokeep]
        Ypost1_pr = Ypost1[:, tokeep]
        if tokeep.any():
            pr[i], _, _, _ = precision_recall_fscore_support(Ytrue_pr, (Ypost1_pr >= t).astype(int))
            pr_avg[i] = np.mean(pr[i])
            coverage[i] = len(pr[i])/len(rc[i])

        ff[i] = (2 * pr_avg[i] * rc_avg[i]) / (pr_avg[i] + rc_avg[i])

    return np.nanmax(ff), coverage[np.argmax(ff)], thresholds[np.argmax(ff)]

def fmax_threshold(Ytrue, Ypost1, t):
    Ytrue = Ytrue.transpose()
    Ypost1 = Ypost1.transpose()
    tokeep = np.where(np.sum(Ytrue, 0) > 0)[0]
    Ytrue = Ytrue[:, tokeep]
    Ypost1 = Ypost1[:, tokeep]

    _, rc, _, _ = precision_recall_fscore_support(Ytrue, (Ypost1 >= t).astype(int))
    rc_avg = np.mean(rc)

    tokeep = np.where(np.sum((Ypost1 >= t).astype(int), 0) > 0)[0]
    Ytrue_pr = Ytrue[:, tokeep]
    Ypost1_pr = Ypost1[:, tokeep]
    if tokeep.any():
        pr, _, _, _ = precision_recall_fscore_support(Ytrue_pr, (Ypost1_pr >= t).astype(int))
        pr_avg = np.mean(pr)
        coverage = len(pr) / len(rc)
    else:
        pr_avg = 0
        coverage = 0

    ff = (2 * pr_avg * rc_avg) / (pr_avg + rc_avg)

    return ff, coverage



aucLin = np.zeros((len(Cs), ))
for i, C in enumerate(Cs):

    with open(modelPath + '/c%s_LR.pkl' % C, 'rb') as fw:
        data = pickle.load(fw)
    clf = data['trained_model']

    Ypost = clf.predict_proba(Xvalid[:, 0:num_fea])

    Ypost1 = np.zeros((Yvalid.shape))
    for j, pred in enumerate(Ypost):
        Ypost1[:, j] = pred[:, 1]

    aucLin[i] = np.nanmean(roc_auc_score(Yvalid, Ypost1, average=None))

    sys.stdout.flush()

print(aucLin)
maxi = np.argmax(aucLin)


with open('../cross_species/Seeds_random_state', 'rb') as f:
    seeds = pickle.load(f)

with open(testsetFile, 'rb') as fw:
    Xtest, Ytest, GO_terms = pickle.load(fw)


bestC = Cs[maxi]

with open(modelPath + 'c%s_LR.pkl' % bestC, 'rb') as fw:
    data = pickle.load(fw)
clf = data['trained_model']

Ypost = clf.predict_proba(Xtest)
Ypost1 = np.zeros((Ytest.shape))
for jj, pred in enumerate(Ypost):
    Ypost1[:, jj] = pred[:, 1]

with open(modelPath + '/predictions.pkl', 'wb') as fw:
    pickle.dump({'Yval': Ytest, 'Ypost': Ypost1}, fw)

aucLin = np.nanmean(roc_auc_score(Ytest, Ypost1, average=None))
avg_fmax, cov, threshold = fmax(Ytest, Ypost1)

with open(modelPath + '/performance.pkl', 'wb') as fw:
    pickle.dump({'tc': aucLin, 'pc': avg_fmax}, fw)

boot_results_f1 = np.zeros(len(seeds))
boot_results_rocauc = np.zeros(len(seeds))

for i, seed in enumerate(seeds):
    Ypost_b, Yval_b = resample(Ypost1, Ytest, random_state=seed, stratify=Ytest)
    boot_results_rocauc[i] = np.nanmean(roc_auc_score(Yval_b, Ypost_b, average=None))
    fmea = fmax_threshold(Yval_b, Ypost_b, threshold)
    boot_results_f1[i] = fmea[0]

print('rocauc')
print(aucLin)
print(np.percentile(boot_results_rocauc, [2.5, 97.5]))

print('f1-score: %s' % avg_fmax)
print('coverage: %s' % cov)
print('threshold: %s' % threshold)
print('confidence interval')
print(np.percentile(boot_results_f1, [2.5, 97.5]))

sys.stdout.flush()
