from numpy import load
import numpy as np
import pickle
import sys

# This script extracts the protein-level embeddings from the .npz file
# that is the output of the SeqVec model

def embeddings_aa(species):

    boo = False
    embedding = []
    item = []
    data = load(seqvecOutputPath + '/Embeddings_%s.npz' % species)
    with open(fastaPath + '/%s_proteinsequence.fasta' % species) as f:
        for line in f:
            if line[0] == '>':
                if boo:
                    embedding.append(data[protein_id])
                    item.append(protein_id)
                boo = True
                protein_id = line.split('|')[1]
        embedding.append(data[protein_id])
        item.append(protein_id)

    X = np.array(embedding)
    id_X = dict(zip(item, X))

    with open(embeddingsPath + '/X%s' % species, 'wb') as f:
        pickle.dump(id_X, f)

    print('done %s' % species)
    return id_X


fastaPath = sys.argv[1]
seqvecOutputPath = sys.argv[2]
embeddingsPath = sys.argv[3]


embedding_aa_rat = embeddings_aa('rat')
embedding_aa_mouse = embeddings_aa('mouse')
embedding_aa_celegans = embeddings_aa('c.elegans')
embedding_aa_yeast = embeddings_aa('yeast')
embedding_aa_human = embeddings_aa('human')
embedding_aa_zebrafish = embeddings_aa('zebrafish')
embedding_aa_athaliana = embeddings_aa('a.thaliana')
