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

Cs = [1e-3, 1e-2, 1e-1, 1, 10, 100]
# Fea = [3072, 5120, 10240]
Fea = [10240]

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/data/XYdata_genmean_scaled_valid_baseline', 'rb') as fw:
    Xvalid, Yvalid, GO_terms = pickle.load(fw)
print('from baseline')

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



aucLin = np.zeros((len(Cs), len(Fea)))
for i, C in enumerate(Cs):
    for ii, num_fea in enumerate(Fea):
        with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/generalized_means/Trained_LR/c%s_LR_numfea%s_1024LSTM1_baseline.pkl' % (C, num_fea), 'rb') as fw:
            data = pickle.load(fw)
        clf = data['trained_model']

        Ypost = clf.predict_proba(Xvalid[:, 0:num_fea])

        with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/generalized_means/Trained_LR/c%s_LR_numfea%s_1024LSTM1_valid_baseline.pkl' % (C, num_fea), 'wb') as fw:
            pickle.dump({'Yval': Yvalid, 'Ypost': Ypost}, fw)

        Ypost1 = np.zeros((Yvalid.shape))
        for j, pred in enumerate(Ypost):
            Ypost1[:, j] = pred[:, 1]

        aucLin[i, ii] = np.nanmean(roc_auc_score(Yvalid, Ypost1, average=None))

        sys.stdout.flush()

print(aucLin)
maxi = np.argmax(aucLin, axis=0)
print('best C values')
print('for 3072: %s' % Cs[maxi[0]])
# print('for 5120: %s' % Cs[maxi[1]])
# print('for 10240: %s' % Cs[maxi[2]])

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/Seeds_random_state', 'rb') as f:
    seeds = pickle.load(f)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/data/XYdata_genmean_scaled_test', 'rb') as fw:
    Xtest, Ytest, GO_terms = pickle.load(fw)

for j, num_fea in enumerate(Fea):
    bestC = Cs[maxi[j]]
    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/generalized_means/Trained_LR/c%s_LR_numfea%s_1024LSTM1_baseline.pkl' % (bestC, num_fea), 'rb') as fw:
        data = pickle.load(fw)
    clf = data['trained_model']

    Ypost = clf.predict_proba(Xtest[:, 0:num_fea])
    Ypost1 = np.zeros((Ytest.shape))
    for jj, pred in enumerate(Ypost):
        Ypost1[:, jj] = pred[:, 1]

    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/generalized_means/Trained_LR/c%s_LR_numfea%s_1024LSTM1_test_baseline.pkl' % (bestC, num_fea), 'wb') as fw:
        pickle.dump({'Yval': Ytest, 'Ypost': Ypost1}, fw)

    aucLin = np.nanmean(roc_auc_score(Ytest, Ypost1, average=None))
    avg_fmax, cov, threshold = fmax(Ytest, Ypost1)

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





