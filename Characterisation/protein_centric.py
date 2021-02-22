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

bestC = 1

with open('/somedirectory/XYdata_moment_test', 'rb') as fz:
    Xtest, Ytest = pickle.load(fz)

terms2keep = np.load('/somedirectory/termIndicesToUse.npy')

Xtest = Xtest[:, np.r_[0:1024]]
Ytest = Ytest[:, terms2keep]

with open('/somedirectory/c%s_termsall_fea%s.pkl' % (bestC, Xtest.shape[1]), 'rb') as fw:
    data = pickle.load(fw)
clf = data['trained_model']

Ypost = clf.predict_proba(Xtest)
Ypost_adp = np.zeros((Ytest.shape))
for i, pred in enumerate(Ypost):
    Ypost_adp[:, i] = pred[:, 1]


def len_aa(file_names_id):
    id = np.loadtxt('/somedirectory/lists/' + file_names_id, dtype=str).tolist()
    length = np.zeros((len(id), ))
    type = '.pkl'
    for i, x in enumerate(id):
        with open('/somedirectory/dataset/' + x + type, 'rb') as f:
            protein_info = pickle.load(f)
        length[i] = len(protein_info['sequence'])
    return length

len_pro = len_aa('test_final.names')
with open('/somedirectoryt/length.pkl', 'wb') as fw:
    pickle.dump(len_pro, fw)





#########################
with open('/somedirecotry/c%s_termsall_fea%s.pkl' % (bestC, 1024), 'rb') as fw:
    data = pickle.load(fw)
Ytest = data['Ytest']
Ypost_adp = data['Ypost']




num_annotations = np.sum(Ytest, axis=0)
num_annotations_protein = np.sum(Ytest, axis=1)

def fmax_per_pro(Ytrue, Ypost1, threshold):
    Ytrue = Ytrue.transpose()
    Ypost1 = Ypost1.transpose()
    tokeep = np.where(np.sum(Ytrue, 0) > 0)[0]
    Ytrue = Ytrue[:, tokeep]
    Ypost1 = Ypost1[:, tokeep]


    _ , rc, _ , _ = precision_recall_fscore_support(Ytrue, (Ypost1 >= threshold).astype(int))
    rc_avg = np.mean(rc)

    tokeep1 = np.where(np.sum((Ypost1 >= threshold).astype(int), 0) > 0)[0]
    Ytrue_pr = Ytrue[:, tokeep1]
    Ypost1_pr = Ypost1[:, tokeep1]

    pr, _, _, _ = precision_recall_fscore_support(Ytrue_pr, (Ypost1_pr >= threshold).astype(int))
    pr_avg = np.mean(pr)
    coverage = len(pr)/len(rc)

    ff_avg = (2 * pr_avg * rc_avg) / (pr_avg + rc_avg)
    new_rc = rc[tokeep1]
    ff = (2 * pr * new_rc) / (pr + new_rc)
    ff = np.nan_to_num(ff)

    return ff_avg, ff, coverage, tokeep, tokeep1, rc[tokeep1], pr


ff_avg, f1, cov, keep1, keep2, rc, pr = fmax_per_pro(Ytest, Ypost_adp, 1)
ff_avg_im, f1_im, cov_im, keep1_im, keep2_im, rc_im, pr_im = fmax_per_pro(Ytest_im, Ypost_adp_im, 0.34)


hist = np.digitize(num_annotations_protein, [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

df1 = pd.DataFrame({'Annotations (#)': hist.astype(int), 'F1 score': f1_im[keep2] - f1, 'Recall': rc, 'Precision': pr})

def plot(type):
    now = datetime.now()
    current_time = now.strftime("%d%m%Y%H%M%S")
    ######### performance per protein, grouped on protein length
    sns.set(style="whitegrid")
    sns.swarmplot(x="Annotations (#)", y=type, zorder=1, data=df1,
                  size=3)
    plt.title('LR classifier term-centric performance')
    sns.boxplot(x="Annotations (#)", y=type, data=df1, boxprops={'facecolor': 'None', "zorder": 10},
                zorder=10, showfliers=False)
    # plt.ylim([-0.1, 1.1])
    plt.savefig('beeswarm_%s_annotations_1024' % type + current_time + '.png')
    plt.savefig('beeswarm_%s_annotations_1024' % type + current_time + '.pdf')
    plt.close()

plot('F1 score')








len_pro = len_pro[keep1]
len_pro = len_pro[keep2]

hist = np.digitize(len_pro, [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1001])

df = pd.DataFrame({'Protein length (# amino acids)': hist.astype(int), 'F1 score': f1_im[keep2] - f1, 'Recall': rc, 'Precision': pr})

def plot(type):
    now = datetime.now()
    current_time = now.strftime("%d%m%Y%H%M%S")
    ######### performance per protein, grouped on protein length
    sns.set(style="whitegrid")
    sns.swarmplot(x="Protein length (# amino acids)", y=type, zorder=1, data=df,
                  size=3)
    # plt.title('LR classifier term-centric performance')
    sns.boxplot(x="Protein length (# amino acids)", y=type, data=df, boxprops={'facecolor': 'None', "zorder": 10},
                zorder=10, showfliers=False)
    # plt.ylim([-0.1, 1.1])
    plt.savefig('beeswarm_%s_length_change' % type + current_time + '.png')
    plt.savefig('beeswarm_%s_length_change' % type + current_time + '.pdf')
    plt.close()

plot('F1 score')
plot('Recall')
plot('Precision')


########## number of protein annotations vs protein-centric performance
num_annotations_protein = num_annotations_protein[keep1]
num_annotations_protein = num_annotations_protein[keep2]

# hist_numanno = np.digitize(num_annotations_protein, [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 42])
hist_numanno = np.digitize(num_annotations_protein, [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 42])


df2 = pd.DataFrame({'Protein annotations (#)': hist_numanno.astype(int), 'F1 score': f1_im[keep2] - f1, 'Recall': rc, 'Precision': pr})

def plot(type):
    now = datetime.now()
    current_time = now.strftime("%d%m%Y%H%M%S")
    ######### performance per protein, grouped on protein length
    sns.set(style="whitegrid")
    sns.swarmplot(x="Protein annotations (#)", y=type, zorder=1, data=df2,
                  size=3)  # , order=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    # plt.title('LR classifier term-centric performance')
    sns.boxplot(x="Protein annotations (#)", y=type, data=df2, boxprops={'facecolor': 'None', "zorder": 10},
                zorder=10, showfliers=False)
    # plt.ylim([-0.1, 1.1])
    plt.savefig('beeswarm_%s_numanno_ELMo_1024' % type + current_time + '.png')
    plt.savefig('beeswarm_%s_numanno_ELMo_1024' % type + current_time + '.pdf')
    plt.close()

plot('F1 score')
plot('Recall')
plot('Precision')


 

# correlation length and number of annotations
stats.spearmanr(len_pro, num_annotations_protein)

from scipy import stats
stats.spearmanr(len_pro, f1)
stats.spearmanr(f1, num_annotations_protein)





