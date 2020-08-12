import numpy as np
from numpy import load
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.stats import moment

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/data/Y_seq_geneNames.pkl', 'rb') as f:
    Ydict = pickle.load(f)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/data/termNames_seq.pkl', 'rb') as f:
    term = pickle.load(f)
    term = [item.decode('utf-8') for item in term]

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/data/termNames_seq_filtered.pkl', 'rb') as f:
    term_filt = pickle.load(f)
term_filt = np.array(term_filt)
pos_filt = [term.index(i) for i in term_filt]



def embeddings_aa(file_names_id):
    item = []
    ### let op, change deze directory nog
    data = load('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/seqvec_embedder/embeddings-3k-bigdata.pkl.npz')
    id = np.loadtxt('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/'
                    'sequence-only/lists/' + file_names_id, dtype=str).tolist()
    embedding_flat = np.zeros((len(id), 10240))
    Y = np.zeros((len(id), 685))
    #get only lstm1 layer
    layer = 1
    for ii, x in enumerate(id):
        a = data[x][layer, :, :]
        b = np.mean(a, axis=0)
        c = moment(a, moment=[2, 3, 4, 5, 6, 7, 8, 9, 10])
        d = np.array([b, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]])
        flat = d.reshape(1024 * 10, order='C').astype(float)
        embedding_flat[ii, :] = flat
        item.append(x)
        Y[ii] = Ydict[x].toarray().reshape(-1)[pos_filt]

    item = np.array(item)
    return embedding_flat, Y, item


X_test, Y_test, id_test = embeddings_aa('test_final.names')
X_valid, Y_valid, id_valid = embeddings_aa('valid_final.names')
X_train, Y_train, id_train = embeddings_aa('train_final.names')


terms2keep = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/lists/termIndicesToUse.npy')

Y_train = Y_train[:, terms2keep]
Y_valid = Y_valid[:, terms2keep]
Y_test = Y_test[:, terms2keep]
GO_terms = term_filt[terms2keep]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/data/XYdata_moments_scaled_train', 'wb') as fw:
    pickle.dump([X_train, Y_train, GO_terms], fw, protocol=4)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/data/XYdata_moments_scaled_valid', 'wb') as fx:
    pickle.dump([X_valid, Y_valid, GO_terms], fx, protocol=4)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/data/XYdata_moments_scaled_test', 'wb') as fz:
    pickle.dump([X_test, Y_test, GO_terms], fz, protocol=4)
