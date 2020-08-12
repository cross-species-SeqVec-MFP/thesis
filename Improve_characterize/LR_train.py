import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
import sys

Cvalue = [[1e-3], [1e-1], [10], [1e-2], [1], [100]]
Cs = Cvalue[int(sys.argv[1])]
num_fea = int(sys.argv[2]) #3072, 5120 or 10240

#this data is scaled already
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/data/XYdata_genmean_scaled_train', 'rb') as fw:
    X_train, Y_train, GO_terms = pickle.load(fw)
print('from baseline')

X_train = X_train[:, 0:num_fea]

C = Cs[0]
print(C)
print('features: %s' % num_fea)
sys.stdout.flush()

clf = MultiOutputClassifier(SGDClassifier(loss='log', penalty='l2', alpha=C, max_iter=1000, random_state=0))
clf.fit(X_train, Y_train)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/generalized_means/Trained_LR/c%s_LR_numfea%s_1024LSTM1.pkl' % (C, num_fea), 'wb') as fw:
    pickle.dump({'trained_model': clf}, fw, protocol=4)
