from numpy import load
import numpy as np
import pickle
import re
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# this code gives distributions of the sequence identify to the mouse training set

blastpPath = sys.argv[1]

with open('./protein_names.pkl', 'rb') as f:
    proteins = pickle.load(f)


def BLAST_alignment(species, index_query, index_alignment, index_identity, prot):
    """ This function gives the protein id's of the database
     proteins that are aligned to the query proteins"""
    alignments = {}
    seq_id = []
    boo = True
    with open(blastpPath + '/BLAST_%s_mouse' % species) as f:
        for line in f:
            if boo:
                if line[0] != '#':
                    query = re.split("\||\t", line)[index_query]
                    iden = float(re.split("\||\t", line)[index_identity])
                    if query in prot:
                        seq_id.append(iden)
                    boo = False
            if line[0] == '#':
                boo = True

    return np.array(seq_id)


aligned_mouse = BLAST_alignment('mouse', 0, 1, 2, proteins['mouse_test'])
aligned_rat = BLAST_alignment('rat', 1, 3, 4, proteins['rat'])
aligned_human = BLAST_alignment('human', 1, 3, 4, proteins['human'])
aligned_zebrafish = BLAST_alignment('zebrafish', 1, 3, 4, proteins['zebrafish'])
aligned_celegans = BLAST_alignment('celegans', 1, 3, 4, proteins['celegans'])
aligned_yeast = BLAST_alignment('yeast', 1, 3, 4, proteins['yeast'])
aligned_athaliana = BLAST_alignment('athaliana', 1, 3, 4, proteins['athaliana'])

def hist_rocauc(perri, species):
    plt.figure()
    plt.hist(perri, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], weights=np.ones(len(perri)) / len(perri), color='darkorange', alpha=1, rwidth=0.85)
    plt.axvline(perri.mean(), 0, 1, label='pyplot vertical line')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Sequence identity')
    plt.ylabel('Count')
    

hist_rocauc(aligned_mouse, 'mouse_test')
hist_rocauc(aligned_rat, 'rat')
hist_rocauc(aligned_human, 'human')
hist_rocauc(aligned_zebrafish, 'zebrafish')
hist_rocauc(aligned_celegans, 'celegans')
hist_rocauc(aligned_yeast, 'yeast')
hist_rocauc(aligned_athaliana, 'athaliana')
