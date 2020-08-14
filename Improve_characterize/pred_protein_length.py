import numpy as np
import pickle
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import OneHotEncoder
import sys
from sklearn.utils import resample

Cvalue = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]  # for regularization in log reg

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/Scaled_data/XYdata_moment_train', 'rb') as fw:
    Xtrain, Ytrain = pickle.load(fw)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/Scaled_data/XYdata_moment_valid', 'rb') as fx:
    Xvalid, Yvalid = pickle.load(fx)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/Scaled_data/XYdata_moment_test', 'rb') as fz:
    Xtest, Ytest = pickle.load(fz)


Xtrain = Xtrain[:, np.r_[0:1024]]
Xvalid = Xvalid[:, np.r_[0:1024]]
Xtest = Xtest[:, np.r_[0:1024]]

enc = OneHotEncoder(handle_unknown='ignore')
 
def len_aa(file_names_id):
    id = np.loadtxt('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/'
                    'sequence-only/lists/' + file_names_id, dtype=str).tolist()
    length = np.zeros((len(id), ))
    type = '.pkl'
    for i, x in enumerate(id):
        with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/dataset/' + x + type, 'rb') as f:
            protein_info = pickle.load(f)
        length[i] = len(protein_info['sequence'])

    hist = np.digitize(length.reshape(-1,1), [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1001])
    Y = enc.fit_transform(hist).toarray()

    return Y

Y_test = len_aa('test_final.names')
Y_valid = len_aa('valid_final.names')
Y_train = len_aa('train_final.names')


for Cs in Cvalue:
    print(Cs)
    sys.stdout.flush()
    clf = MultiOutputClassifier(SGDClassifier(loss='log', penalty='l2', alpha=Cs, max_iter=1000, random_state=0))

    clf.fit(Xtrain, Y_train)

    Ypost = clf.predict_proba(Xvalid)

    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/predictions/'
              'moments/validation/c%spredict_length.pkl' % Cs, 'wb') as fw:
        pickle.dump({'Yvalid': Y_valid, 'Ypost': Ypost, 'trained_model': clf}, fw)

    sys.stdout.flush()


#############################
# validation results
res = np.zeros(len(Cvalue))
for joe, C in enumerate(Cvalue):
    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/predictions/'
              'moments/validation/c%spredict_length.pkl' % C, 'rb') as fw:
        data = pickle.load(fw)

    Ypost = np.zeros((data['Yvalid'].shape))
    for j, pred in enumerate(data['Ypost']):
        Ypost[:, j] = pred[:, 1]

    res[joe] = np.nanmean(roc_auc_score(data['Yvalid'], Ypost, average=None))

bestC = Cvalue[np.argmax(res)]
print('best C %s' % bestC)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/predictions/'
          'moments/validation/c%spredict_length.pkl' % bestC, 'rb') as fw:
    data = pickle.load(fw)

ccc = data['trained_model']
Ypost = ccc.predict_proba(Xtest)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/predictions/'
          'moments/validation/c%spredict_lengthtest.pkl' % bestC, 'wb') as fw:
    pickle.dump({'Yvalid': Y_test, 'Ypost': Ypost}, fw)

####
bestC=0.0001
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/predictions/'
          'moments/validation/c%spredict_lengthtest.pkl' % bestC, 'rb') as fw:
    data = pickle.load(fw)

Ypost1 = np.zeros((data['Yvalid'].shape))
for jj, pred1 in enumerate(data['Ypost']):
    Ypost1[:, jj] = pred1[:, 1]

print(roc_auc_score(data['Yvalid'], Ypost1, average=None))

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/Seeds_random_state', 'rb') as f:
    seeds = pickle.load(f)
boot_results_rocauc = np.zeros((len(seeds), 10))

for ij, seed in enumerate(seeds):
    Ypost, Yval = resample(Ypost1, data['Yvalid'], random_state=seed, stratify=data['Yvalid'])
    iiii = np.where(np.sum(Yval, 0) > 0)[0]
    boot_results_rocauc[ij,: ] = roc_auc_score(Yval[:, iiii], Ypost[:, iiii], average=None)

print('confidence interval')
for k in range(10):
    print(np.percentile(boot_results_rocauc[:, k], [2.5, 97.5]))
    sys.stdout.flush()



