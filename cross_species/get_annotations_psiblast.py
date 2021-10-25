from numpy import load
import numpy as np
import pickle
import re
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import sys

onto = sys.argv[1]
if onto == 'C':
    type_GO = 'cellular_component'
    root_term = 'GO:0005575'
elif onto == 'P':
    type_GO = 'biological_process'
    root_term = 'GO:0008150'
else:
    type_GO = 'molecular_function'
    root_term = 'GO:0003674'


directory = sys.argv[2]

directory1 = sys.argv[3]

directory2 = sys.argv[4]

directory3 = sys.argv[5]

# this code retrieves the GO annotations from PSI-BLAST alignments

def get_indexes(species):
    with open('%s/index_protein_centric/%s.pkl' % (directory1, species), 'rb') as f:
        protein_indexes = pickle.load(f)
    protein_indexes.remove(root_term)
    with open('%s/index_term_centric/%s.pkl' % (directory1, species), 'rb') as f:
        term_indexes = pickle.load(f)
    term_indexes.remove(root_term)
    return protein_indexes, term_indexes


protein_indexes_human, term_indexes_human = get_indexes('human')
protein_indexes_rat, term_indexes_rat = get_indexes('rat')
protein_indexes_mouse_test, term_indexes_mouse_test = get_indexes('mouse_test')
protein_indexes_mouse_valid, term_indexes_mouse_valid = get_indexes('mouse_valid')
protein_indexes_zebrafish, term_indexes_zebrafish = get_indexes('zebrafish')
protein_indexes_celegans, term_indexes_celegans = get_indexes('celegans')
protein_indexes_yeast, term_indexes_yeast = get_indexes('yeast')
protein_indexes_athaliana, term_indexes_athaliana = get_indexes('athaliana')

with open('%s/mouse_index_traintestvalid.pkl' % directory, 'rb') as f:
    mouse_train_ind, mouse_validtest_ind, mouse_valid_ind, mouse_test_ind = pickle.load(f)

def open_input(species):
    with open('%s/%s.pkl' % (directory, species), 'rb') as f:
        yspecies, Xspecies, proteins, loc_species = pickle.load(f)
    return Xspecies, yspecies, loc_species, proteins

Xmouse, ymouse, loc_mouse, prot_mouse = open_input('mouse')
inv_loc_mouse = {v: k for k, v in loc_mouse.items()}
prot_mousevalidtest = prot_mouse[mouse_validtest_ind]
prot_mouse_test = prot_mousevalidtest[mouse_test_ind]
prot_mouse_valid = prot_mousevalidtest[mouse_valid_ind]

Xrat, yrat, loc_rat, prot_rat = open_input('rat')
Xhuman, yhuman, loc_human, prot_human = open_input('human')
Xzebrafish, yzebrafish, loc_zebrafish, prot_zebrafish = open_input('zebrafish')
Xcelegans, ycelegans, loc_celegans, prot_celegans = open_input('celegans')
Xyeast, yyeast, loc_yeast, prot_yeast = open_input('yeast')
Xathaliana, yathaliana, loc_athaliana, prot_athaliana = open_input('athaliana')

with open('%s/mouse_index_setsY.pkl' % directory, 'rb') as f:
    ymouse_train, ymouse_valid, ymouse_test = pickle.load(f)

############################################################################
# Getting protein id's of aligned proteins

def BLAST_alignment(species, index_query, index_alignment, index_identity, prot):
    """ This function gives the protein id's of the database
     proteins that are aligned to the query proteins"""
    alignments = {}
    boo = False
    with open('%s/psiBLAST_%s' % (directory2, species)) as f:
        for line in f:
            if line[0] != '#' and line != '\n' and line != 'Search has CONVERGED!\n':
                aligned.append(re.split("\||\t", line)[index_alignment])
                identity.append(re.split("\||\t", line)[index_identity])
                query = re.split("\||\t", line)[index_query]
                boo = True
            if line[0] == '#':
                if boo:
                    boo = False
                    if query in prot:
                        alignments[query] = np.column_stack((aligned, identity))

                aligned = []
                identity = []

    print('number aligned for %s: %s' % (species, len(alignments)))

    return alignments

#
aligned_mousevalid = BLAST_alignment('mouse_valid', 0, 1, 2, prot_mouse_valid)
aligned_mouse = BLAST_alignment('mouse_test', 0, 1, 2, prot_mouse_test)
aligned_rat = BLAST_alignment('rat', 1, 3, 4, prot_rat)
aligned_human = BLAST_alignment('human', 1, 3, 4, prot_human)
aligned_zebrafish = BLAST_alignment('zebrafish', 1, 3, 4, prot_zebrafish)
aligned_celegans = BLAST_alignment('celegans', 1, 3, 4, prot_celegans)
aligned_yeast = BLAST_alignment('yeast', 1, 3, 4, prot_yeast)
aligned_athaliana = BLAST_alignment('athaliana', 1, 3, 4, prot_athaliana)

with open('%s/GO_levels_all_GOterms.pkl' % directory1, 'rb') as fw:
    depth_terms = pickle.load(fw)
del depth_terms[root_term]

max_value = 0
terms_per_depth = {k: [] for k in range(20)}
for key in depth_terms.keys():
    depth = depth_terms[key][0]
    terms_per_depth[depth].append(key)
    if depth > max_value:
        max_value = depth


def assinging_Y(species, aligned, prot, dict_index, yvalid, location):

    Ypost = np.zeros((yvalid.shape))

    for key in aligned.keys():
        for ali_prot, identity in enumerate(aligned[key][:, 1]):
            hits_protein = ymouse[np.where(prot_mouse == aligned[key][ali_prot, 0])[0], :]
            terms_to_assign = [inv_loc_mouse[index] for index in np.where(hits_protein == 1)[1]]
            for term1 in terms_to_assign:
                if term1 in location:
                    if float(identity)/100 >= Ypost[np.where(prot == key)[0], location[term1]]:
                        Ypost[np.where(prot == key)[0], location[term1]] = float(identity)/100

    for depth1 in np.arange(max_value, -1, -1):
        for key1 in dict_index:
            if key1 in terms_per_depth[depth1]:
                if depth_terms[key1][0]:
                    for up_terms in depth_terms[key1]:
                        if up_terms in dict_index:
                            index = np.where(Ypost[:, location[key1]] < Ypost[:, location[up_terms]])[0]
                            if index.any():
                                for protein in index:
                                    Ypost[protein, location[key1]] = Ypost[protein, location[up_terms]]


    with open('%s/psiblast_%s.pkl' % (directory3, species), 'wb') as fw:
        pickle.dump({'Yval': yvalid, 'Ypost': Ypost}, fw)

    return yvalid, Ypost


mouse_valid = assinging_Y('mouse_valid', aligned_mousevalid, prot_mouse_valid, protein_indexes_mouse_valid, ymouse_valid, loc_mouse)
mouse = assinging_Y('mouse_test', aligned_mouse, prot_mouse_test, protein_indexes_mouse, ymouse_test, loc_mouse)
rat = assinging_Y('rat', aligned_rat, prot_rat, protein_indexes_rat, yrat, loc_rat)
human = assinging_Y('human', aligned_human, prot_human, protein_indexes_human, yhuman, loc_human)
zebrafish = assinging_Y('zebrafish', aligned_zebrafish, prot_zebrafish, protein_indexes_zebrafish, yzebrafish, loc_zebrafish)
celegans = assinging_Y('celegans', aligned_celegans, prot_celegans, protein_indexes_celegans, ycelegans, loc_celegans)
yeast = assinging_Y('yeast', aligned_yeast, prot_yeast, protein_indexes_yeast, yyeast, loc_yeast)
athaliana = assinging_Y('athaliana', aligned_athaliana, prot_athaliana, protein_indexes_athaliana, yathaliana, loc_athaliana)





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

    _ , rc, _ , _ = precision_recall_fscore_support(Ytrue, (Ypost1 >= t).astype(int))
    rc_avg = np.mean(rc)

    tokeep = np.where(np.sum((Ypost1 >= t).astype(int), 0) > 0)[0]
    Ytrue_pr = Ytrue[:, tokeep]
    Ypost1_pr = Ypost1[:, tokeep]
    if tokeep.any():
        pr, _, _, _ = precision_recall_fscore_support(Ytrue_pr, (Ypost1_pr >= t).astype(int))
        pr_avg = np.mean(pr)
        coverage = len(pr)/len(rc)
    else:
        pr_avg = 0
        coverage = 0

    ff = (2 * pr_avg * rc_avg) / (pr_avg + rc_avg)

    return ff, coverage

def data(species):
    with open('%s/psiblast_%s.pkl' % (directory3, species), 'rb') as fw:
        data1 = pickle.load(fw)
    return data1


mouse_perfvalid = fmax(mouse_valid['Yval'], mouse_valid['Ypost'])
print('mouse validation set fmax average: %s, coverage %s threshold %s' % (mouse_perfvalid[0], mouse_perfvalid[1], mouse_perfvalid[2]))

def bootstrap_fmax(Yval1, Ypost1, threshold1):
    boot_results_f1 = np.zeros(len(seeds))

    for ij, seed in enumerate(seeds):
        Ypost, Yval = resample(Ypost1, Yval1, random_state=seed, stratify=Yval1)
        fmea = fmax_threshold(Yval, Ypost, threshold1)
        boot_results_f1[ij] = fmea[0]

    print('confidence interval')
    print(np.percentile(boot_results_f1, [2.5, 97.5]))


mouse_perf = fmax_threshold(mouse[0], mouse[1], mouse_perfvalid[2])
print('mouse fmax average: %s, coverage %s' % (mouse_perf[0], mouse_perf[1]))
bootstrap_fmax(mouse[0], mouse[1], mouse_perfvalid[2])
rat_perf = fmax_threshold(rat[0], rat[1], mouse_perfvalid[2])
print('rat fmax average: %s, coverage %s' % (rat_perf[0], rat_perf[1]))
bootstrap_fmax(rat[0], rat[1], mouse_perfvalid[2])
human_perf = fmax_threshold(human[0], human[1], mouse_perfvalid[2])
print('human fmax average: %s, coverage %s' % (human_perf[0], human_perf[1]))
bootstrap_fmax(human[0], human[1], human_perf[2])
zebrafish_perf = fmax_threshold(zebrafish[0], zebrafish[1], mouse_perfvalid[2])
print('zebrafish fmax average: %s, coverage %s' % (zebrafish_perf[0], zebrafish_perf[1]))
bootstrap_fmax(zebrafish[0], zebrafish[1], zebrafish_perf[2])
celegans_perf = fmax_threshold(celegans[0], celegans[1], mouse_perfvalid[2])
print('celegans fmax average: %s, coverage %s' % (celegans_perf[0], celegans_perf[1]))
bootstrap_fmax(celegans[0], celegans[1], celegans_perf[2])
yeast_perf = fmax_threshold(yeast[0], yeast[1], mouse_perfvalid[2])
print('yeast fmax average: %s, coverage %s' % (yeast_perf[0], yeast_perf[1]))
bootstrap_fmax(yeast[0], yeast[1], yeast_perf[2])
athaliana_perf = fmax_threshold(athaliana[0], athaliana[1], mouse_perfvalid[2])
print('athaliana fmax average: %s, coverage %s' % (athaliana_perf[0], athaliana_perf[1]))
bootstrap_fmax(athaliana[0], athaliana[1], athaliana_perf[2])

#
# ######################## term centric
#
def rocauc(dict_index, species, predictions, location):

    Yval = np.zeros((predictions[0].shape[0], len(dict_index)))
    Ypost = np.zeros((predictions[0].shape[0], len(dict_index)))

    for i_key, key in enumerate(dict_index):
        Yval[:, i_key] = predictions[0][:, location[key]]
        Ypost[:, i_key] = predictions[1][:, location[key]]

    aucLin1 = np.nanmean(roc_auc_score(Yval, Ypost, average=None))
    print('%s rocauc average: %s' % (species, aucLin1))

    boot_results_rocauc = np.zeros(len(seeds))

    for ij, seed in enumerate(seeds):
        Ypost1, Yval1 = resample(Ypost, Yval, random_state=seed, stratify=Yval)
        iiii = np.where(np.sum(Yval1, 0) > 0)[0]
        boot_results_rocauc[ij] = np.nanmean(roc_auc_score(Yval1[:, iiii], Ypost1[:, iiii], average=None))

    print('confidence interval')
    print(np.percentile(boot_results_rocauc, [2.5, 97.5]))


rocauc(term_indexes_mouse_test, 'mouse_test', mouse, loc_mouse)
rocauc(term_indexes_rat, 'rat', rat, loc_rat)
rocauc(term_indexes_human, 'human', human, loc_human)
rocauc(term_indexes_zebrafish, 'zebrafish', zebrafish, loc_zebrafish)
rocauc(term_indexes_celegans, 'celegans', celegans, loc_celegans)
rocauc(term_indexes_yeast, 'yeast', yeast, loc_yeast)
rocauc(term_indexes_athaliana, 'athaliana', athaliana, loc_athaliana)
