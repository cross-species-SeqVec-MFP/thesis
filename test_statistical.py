import numpy as np
import pickle
import sys
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

def opens(start_index, end_index, type, feat):
    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/%s/Predictions/predictions_term_centric_NNfea%s:%s%s.pkl' % (feat, start_index, end_index, type), 'rb') as fw:
        data = pickle.load(fw)
    Ytrue = data['Yval']
    Ypred = data['Ypost']
    rocauc = roc_auc_score(Ytrue, Ypred, average=None)
    return rocauc

roc_baseline = opens(0, 1024, '', 'layers')

def scatter_change(start_index, end_index, type, feat):
    now = datetime.now()
    current_time = now.strftime("%d%m%Y%H%M%S")

    rocauc = opens(start_index, end_index, type, feat)

    if type == '':
        type = 'lstm1'

    print(type)
    print(feat)
    print('%s:%s' % (start_index, end_index))
    print('ttest')
    print(stats.ttest_1samp(rocauc - roc_baseline, 0))

    plt.hlines(np.mean(rocauc - roc_baseline), 0.4, 1)
    plt.scatter(roc_baseline, rocauc - roc_baseline, s=2, color='darkorange', alpha=0.75)
    plt.grid(alpha=0.75)
    plt.xlabel('Rocauc score baseline')
    plt.ylabel('Change in rocauc performance')
    plt.xlim((0.4, 1))
    plt.ylim((-0.18, 0.18))
    plt.title('Change in performance MLP\n'
              '%s:%s %s %s features' % (start_index, end_index, type, feat))

    plt.savefig('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/Figures/scatter_change_rocauc__%s%sfea_%s_%s' % (start_index, end_index, type, feat) + current_time + '.pdf')
    plt.close()




scatter_change(0, 3072, '_baseline', 'generalized_means')
scatter_change(0, 5120, '_baseline', 'generalized_means')
scatter_change(0, 10240, '_baseline', 'generalized_means')

scatter_change(0, 3072, '', 'generalized_means')
scatter_change(0, 5120, '', 'generalized_means')
scatter_change(0, 10240, '', 'generalized_means')

scatter_change(0, 3072, '_baseline', 'moments')
scatter_change(0, 5120, '_baseline', 'moments')
scatter_change(0, 10240, '_baseline', 'moments')

scatter_change(0, 3072, '', 'moments')
scatter_change(0, 5120, '', 'moments')
scatter_change(0, 10240, '', 'moments')

scatter_change(0, 3072, '', 'layers')
scatter_change(1024, 2048, '', 'layers')
scatter_change(2048, 3072, '', 'layers')



################## for the last LR related things
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/predictions/gen_mean/test/results.pkl', 'rb') as fw:
    perf1024, perf3072, perf5120, perf10240, terms = pickle.load(fw)

ave_term_1024 = np.mean(perf1024, axis=1)

bestC=1


def opens(num_fea, type, feat):
    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/%s/Trained_LR/c%s_LR_numfea%s_1024LSTM1_test%s.pkl' % (feat, bestC, num_fea, type), 'rb') as fw:
        data = pickle.load(fw)
    Ytrue = data['Yval']
    Ypred = data['Ypost']
    rocauc = roc_auc_score(Ytrue, Ypred, average=None)
    return rocauc


def scatter_change(num_fea, type, feat):
    now = datetime.now()
    current_time = now.strftime("%d%m%Y%H%M%S")

    rocauc = opens(num_fea, type, feat)

    if type == '':
        type = 'lstm1'

    print(type)
    print(feat)
    print('%s' % (num_fea))
    print('ttest')
    print(stats.ttest_1samp(rocauc - ave_term_1024, 0))

    plt.hlines(np.mean(rocauc - ave_term_1024), 0.4, 1)
    plt.scatter(ave_term_1024, rocauc - ave_term_1024, s=2, color='darkorange', alpha=0.75)
    plt.grid(alpha=0.75)
    plt.xlabel('Rocauc score baseline')
    plt.ylabel('Change in rocauc performance')
    plt.xlim((0.4, 1))
    plt.ylim((-0.18, 0.18))
    plt.title('Change in performance LR\n'
              '%s %s %s features' % (num_fea, type, feat))

    plt.savefig('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/Figures/scatter_change_rocauc__LR%sfea_%s_%s' % (num_fea, type, feat) + current_time + '.pdf')
    plt.close()




scatter_change(10240, '_baseline', 'generalized_means')

scatter_change(3072, '', 'generalized_means')
scatter_change(5120, '', 'generalized_means')
scatter_change(10240, '', 'generalized_means')

scatter_change(3072, '', 'moments')
scatter_change(5120, '', 'moments')
scatter_change(10240, '', 'moments')



####################### bonferoni correction p-values
from statsmodels.sandbox.stats.multicomp import multipletests

p_LR = [6.24e-14, 2.94e-27, 2.59e-6,
        5.98e-26, 9.51e-28, 5.19e-28,
        2.45e-9, 3.01e-13, 1.36e-17,
        5.81e-19, 1.80e-29, 1.67e-36,
        7.52e-9, 1.48e-17, 6.03e-4]

adjustedp_LR = multipletests(p_LR, alpha=0.05, method='bonferroni')

p_MLP = [0.749, 0.878, 6.63e-3,
         0.726, 2.48e-8, 6.03e-9,
         0.333, 3.20e-3, 9.84e-6,
         0.633, 0.407, 8.88e-4,
         0.294, 0.0388, 8.39e-11]

adjustedp_MLP = multipletests(p_MLP, alpha=0.05, method='bonferroni')





