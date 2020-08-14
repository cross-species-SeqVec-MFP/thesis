from numpy import load
import numpy as np
import pickle
from sklearn.metrics import matthews_corrcoef
from statistics import mean


############################################################################
# Getting protein id's of aligned proteins
blast = {}
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/blast/blastout_test_tophit.txt') as f:
    for line in f:
        if line[0] != '#' and line[0] != '-':
            blast[line.split("\t")[0]] = line.split("\t")[1]


with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/data/Y_seq_geneNames.pkl', 'rb') as f:
    Ydict = pickle.load(f)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/data/termNames_seq.pkl', 'rb') as f:
    term = pickle.load(f)
    term = [item.decode('utf-8') for item in term]

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/data/termNames_seq_filtered.pkl', 'rb') as f:
    term_filt = pickle.load(f)
pos_filt = [term.index(i) for i in term_filt]

terms2keep = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/lists/termIndicesToUse.npy')

names = np.loadtxt('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/lists/test_final.names', dtype=str).tolist()
Ytrue = []
Ytest = []

for p in names:
    if p in blast.keys():
        Ytrue.append(Ydict[p].toarray().reshape(-1)[pos_filt])
        Ytest.append(Ydict[blast[p]].toarray().reshape(-1)[pos_filt])

Ytrue = np.array(Ytrue)
Ytest = np.array(Ytest)
Ytrue = Ytrue[:, terms2keep]
Ytest = Ytest[:, terms2keep]

performance = []
for i in np.arange(0, 441, 1):
    performance.append(matthews_corrcoef(Ytrue[:, i], Ytest[:, i]))

print("BLAST classifier has average mcc performance of %s" % mean(performance))

