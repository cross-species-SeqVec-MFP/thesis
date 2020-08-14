from numpy import load
import numpy as np
import pickle
import re
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/protein_names.pkl', 'rb') as f:
    proteins = pickle.load(f)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_term_centric/mouse.pkl', 'rb') as f:
    mousetrain_indexes = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/mouse_valid.pkl', 'rb') as f:
    mousevalid_indexes = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/mouse_test.pkl', 'rb') as f:
    mousetest_indexes = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/rat.pkl', 'rb') as f:
    rat_indexes = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/yeast.pkl', 'rb') as f:
    yeast_indexes = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/celegans.pkl', 'rb') as f:
    celegans_indexes = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/human.pkl', 'rb') as f:
    human_indexes = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/zebrafish.pkl', 'rb') as f:
    zebrafish_indexes = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/athaliana.pkl', 'rb') as f:
    athaliana_indexes = pickle.load(f)

############################################################################
# Getting protein id's of aligned proteins

def BLAST_alignment(species, index_query, index_alignment, index_identity, prot):
    """ This function gives the protein id's of the database
     proteins that are aligned to the query proteins"""
    alignments = {}
    boo = False
    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/BLAST/BLAST_%s_mouse' % species) as f:
        for line in f:
            if line[0] != '#':
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
# aligned_mousevalid = BLAST_alignment('mousevalid', 0, 1, 2, proteins['mouse_valid'])
# aligned_mouse = BLAST_alignment('mouse', 0, 1, 2, proteins['mouse_test'])
# aligned_rat = BLAST_alignment('rat', 1, 3, 4, proteins['rat'])
aligned_human = BLAST_alignment('human', 1, 3, 4, proteins['human'])
# aligned_zebrafish = BLAST_alignment('zebrafish', 1, 3, 4, proteins['zebrafish'])
# aligned_celegans = BLAST_alignment('celegans', 1, 3, 4, proteins['celegans'])
# aligned_yeast = BLAST_alignment('yeast', 1, 3, 4, proteins['yeast'])
# aligned_athaliana = BLAST_alignment('athaliana', 1, 3, 4, proteins['athaliana'])

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GO_terms/GO_depth_all_mouse_proteins.pkl', 'rb') as fw:
    depth_terms = pickle.load(fw)

max_value = 0
terms_per_depth = {k: [] for k in range(20)}
for key in depth_terms.keys():
    depth = depth_terms[key][0]
    terms_per_depth[depth].append(key)
    if depth > max_value:
        max_value = depth


with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/mouse.pkl', 'rb') as f:
    ymouse, Xm, prot_mouse, loc_mouse = pickle.load(f)
inv_loc_mouse = {v: k for k, v in loc_mouse.items()}

def assinging_Y(species, aligned, prot, dict_index):
    dict_index.remove('GO:0003674')
    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/validation_results/predictions_protein_centric/%s.pkl' % species, 'rb') as fw:
        data = pickle.load(fw)
        Ypost = np.zeros((data['Yval'].shape))

    for key in aligned.keys():
        for ali_prot, identity in enumerate(aligned[key][:, 1]):
            hits_protein = ymouse[np.where(prot_mouse == aligned[key][ali_prot, 0])[0], :]
            terms_to_assign = [inv_loc_mouse[index] for index in np.where(hits_protein == 1)[1]]
            for term1 in terms_to_assign:
                if term1 in data['loc_term']:
                    if float(identity)/100 >= Ypost[np.where(prot == key)[0], data['loc_term'][term1]]:
                        Ypost[np.where(prot == key)[0], data['loc_term'][term1]] = float(identity)/100

    for depth1 in np.arange(max_value, -1, -1):
        for key1 in dict_index:
            if key1 in terms_per_depth[depth1]:
                if depth_terms[key1][1]:
                    for up_terms in depth_terms[key1][1]:
                        if up_terms in dict_index:
                            index = np.where(Ypost[:, data['loc_term'][key1]] < Ypost[:, data['loc_term'][up_terms]])[0]
                            if index.any():
                                for protein in index:
                                    Ypost[protein, data['loc_term'][key1]] = Ypost[protein, data['loc_term'][up_terms]]

    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/BLAST/annotation_result/%s.pkl' % species, 'wb') as fw:
        pickle.dump({'Yval': data['Yval'], 'Ypost': Ypost}, fw)

    return data['Yval'], Ypost


# mouse_valid = assinging_Y('mouse_valid_c%s' % 1e-6, aligned_mousevalid, proteins['mouse_valid'], mousevalid_indexes)
# mouse = assinging_Y('mouse_test', aligned_mouse, proteins['mouse_test'], mousetest_indexes)
# rat = assinging_Y('rat', aligned_rat, proteins['rat'], rat_indexes)
human = assinging_Y('human', aligned_human, proteins['human'], human_indexes)
# zebrafish = assinging_Y('zebrafish', aligned_zebrafish, proteins['zebrafish'], zebrafish_indexes)
# celegans = assinging_Y('celegans', aligned_celegans, proteins['celegans'], celegans_indexes)
# yeast = assinging_Y('yeast', aligned_yeast, proteins['yeast'], yeast_indexes)
# athaliana = assinging_Y('athaliana', aligned_athaliana, proteins['athaliana'], athaliana_indexes)





# def fmax(Ytrue, Ypost1):
#     thresholds = np.linspace(0.0, 1.0, 51)
#     ff = np.zeros(thresholds.shape, dtype=object)
#     pr = np.zeros(thresholds.shape, dtype=object)
#     rc = np.zeros(thresholds.shape, dtype=object)
#     rc_avg = np.zeros(thresholds.shape)
#     pr_avg = np.zeros(thresholds.shape)
#     coverage = np.zeros(thresholds.shape)
#
#     Ytrue = Ytrue.transpose()
#     Ypost1 = Ypost1.transpose()
#     tokeep = np.where(np.sum(Ytrue, 0) > 0)[0]
#     Ytrue = Ytrue[:, tokeep]
#     Ypost1 = Ypost1[:, tokeep]
#
#     for i, t in enumerate(thresholds):
#         _ , rc[i], _ , _ = precision_recall_fscore_support(Ytrue, (Ypost1 >= t).astype(int))
#         rc_avg[i] = np.mean(rc[i])
#
#         tokeep = np.where(np.sum((Ypost1 >= t).astype(int), 0) > 0)[0]
#         Ytrue_pr = Ytrue[:, tokeep]
#         Ypost1_pr = Ypost1[:, tokeep]
#         if tokeep.any():
#             pr[i], _, _, _ = precision_recall_fscore_support(Ytrue_pr, (Ypost1_pr >= t).astype(int))
#             pr_avg[i] = np.mean(pr[i])
#             coverage[i] = len(pr[i])/len(rc[i])
#
#         ff[i] = (2 * pr_avg[i] * rc_avg[i]) / (pr_avg[i] + rc_avg[i])
#
#     return np.nanmax(ff), coverage[np.argmax(ff)], thresholds[np.argmax(ff)]
#
#
# def fmax_threshold(Ytrue, Ypost1, t):
#
#     Ytrue = Ytrue.transpose()
#     Ypost1 = Ypost1.transpose()
#     tokeep = np.where(np.sum(Ytrue, 0) > 0)[0]
#     Ytrue = Ytrue[:, tokeep]
#     Ypost1 = Ypost1[:, tokeep]
#
#     _ , rc, _ , _ = precision_recall_fscore_support(Ytrue, (Ypost1 >= t).astype(int))
#     rc_avg = np.mean(rc)
#
#     tokeep = np.where(np.sum((Ypost1 >= t).astype(int), 0) > 0)[0]
#     Ytrue_pr = Ytrue[:, tokeep]
#     Ypost1_pr = Ypost1[:, tokeep]
#     if tokeep.any():
#         pr, _, _, _ = precision_recall_fscore_support(Ytrue_pr, (Ypost1_pr >= t).astype(int))
#         pr_avg = np.mean(pr)
#         coverage = len(pr)/len(rc)
#     else:
#         pr_avg = 0
#         coverage = 0
#
#     ff = (2 * pr_avg * rc_avg) / (pr_avg + rc_avg)
#
#     return ff, coverage
#
# def data(species):
#     with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/BLAST/annotation_result/%s.pkl' % species, 'rb') as fw:
#         data1 = pickle.load(fw)
#     return data1
#
# mouse_valid = data('mouse_valid_c%s' % 1e-6)
# mouse_perfvalid = fmax(mouse_valid['Yval'], mouse_valid['Ypost'])
# print('mouse validation set fmax average: %s, coverage %s threshold %s' % (mouse_perfvalid[0], mouse_perfvalid[1], mouse_perfvalid[2]))
# mouse_perf = fmax_threshold(mouse[0], mouse[1], mouse_perfvalid[2])
# print('mouse fmax average: %s, coverage %s' % (mouse_perf[0], mouse_perf[1]))
# rat_perf = fmax_threshold(rat[0], rat[1], mouse_perfvalid[2])
# print('rat fmax average: %s, coverage %s' % (rat_perf[0], rat_perf[1]))
# human_perf = fmax_threshold(human[0], human[1], mouse_perfvalid[2])
# print('human fmax average: %s, coverage %s' % (human_perf[0], human_perf[1]))
# zebrafish_perf = fmax_threshold(zebrafish[0], zebrafish[1], mouse_perfvalid[2])
# print('zebrafish fmax average: %s, coverage %s' % (zebrafish_perf[0], zebrafish_perf[1]))
# celegans_perf = fmax_threshold(celegans[0], celegans[1], mouse_perfvalid[2])
# print('celegans fmax average: %s, coverage %s' % (celegans_perf[0], celegans_perf[1]))
# yeast_perf = fmax_threshold(yeast[0], yeast[1], mouse_perfvalid[2])
# print('yeast fmax average: %s, coverage %s' % (yeast_perf[0], yeast_perf[1]))
# athaliana_perf = fmax_threshold(athaliana[0], athaliana[1], mouse_perfvalid[2])
# print('athaliana fmax average: %s, coverage %s' % (athaliana_perf[0], athaliana_perf[1]))
#
#

######################## term centric
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_term_centric/mouse_test.pkl', 'rb') as f:
    mousetest_indexes = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_term_centric/rat.pkl', 'rb') as f:
    rat_indexes = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_term_centric/yeast.pkl', 'rb') as f:
    yeast_indexes = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_term_centric/celegans.pkl', 'rb') as f:
    celegans_indexes = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_term_centric/human.pkl', 'rb') as f:
    human_indexes = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_term_centric/zebrafish.pkl', 'rb') as f:
    zebrafish_indexes = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_term_centric/athaliana.pkl', 'rb') as f:
    athaliana_indexes = pickle.load(f)

def rocauc(dict_index, species):
    dict_index.remove('GO:0003674')
    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/validation_results/predictions_protein_centric/%s.pkl' % species, 'rb') as fw:
        data_loc = pickle.load(fw)

    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/BLAST/annotation_result/%s.pkl' % species, 'rb') as fw:
        data = pickle.load(fw)

    Yval = np.zeros((data['Yval'].shape[0], len(dict_index)))
    Ypost = np.zeros((data['Yval'].shape[0], len(dict_index)))

    for i_key, key in enumerate(dict_index):
        Yval[:, i_key] = data['Yval'][:, data_loc['loc_term'][key]]
        Ypost[:, i_key] = data['Ypost'][:, data_loc['loc_term'][key]]

    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/BLAST/annotation_result/term_centric/%s.pkl' % species, 'wb') as fw:
        pickle.dump({'Yval': Yval, 'Ypost': Ypost}, fw)

    aucLin1 = np.nanmean(roc_auc_score(Yval, Ypost, average=None))
    print('%s rocauc average: %s' % (species, aucLin1))


rocauc(mousetest_indexes, 'mouse_test')
rocauc(rat_indexes, 'rat')
rocauc(human_indexes, 'human')
rocauc(zebrafish_indexes, 'zebrafish')
rocauc(celegans_indexes, 'celegans')
rocauc(yeast_indexes, 'yeast')
rocauc(athaliana_indexes, 'athaliana')
















#
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/mouse_index_traintestvalid.pkl', 'rb') as f:
#     mouse_train_ind, mouse_validtest_ind, mouse_valid_ind, mouse_test_ind = pickle.load(f)
#
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/rat.pkl', 'rb') as f:
#     yrat, Xrat, prot_rat, loc_rat = pickle.load(f)
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/celegans.pkl', 'rb') as f:
#     ycelegans, Xcelegans, prot_celegans, loc_celegans = pickle.load(f)
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/yeast.pkl', 'rb') as f:
#     yyeast, Xyeast, prot_yeast, loc_yeast = pickle.load(f)
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/human.pkl', 'rb') as f:
#     yhuman, Xhuman, prot_human, loc_human = pickle.load(f)
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/zebrafish.pkl', 'rb') as f:
#     yzebrafish, Xzebrafish, prot_zebrafish, loc_zebrafish = pickle.load(f)
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/athaliana.pkl', 'rb') as f:
#     yathaliana, Xathaliana, prot_athaliana, loc_athaliana = pickle.load(f)
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/mouse.pkl', 'rb') as f:
#     ymouse, Xm, prot_mouse, loc_mouse = pickle.load(f)
# prot_mouse_train = prot_mouse[mouse_train_ind]
# prot_mousevalidtest = prot_mouse[mouse_validtest_ind]
# prot_mouse_test = prot_mousevalidtest[mouse_test_ind]
# prot_mouse_valid = prot_mousevalidtest[mouse_valid_ind]
#
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/protein_names.pkl', 'wb') as f:
#     pickle.dump({'mouse_train': prot_mouse_train, 'mouse_valid': prot_mouse_valid, 'mouse_test': prot_mouse_test, 'rat': prot_rat,
#                  'human': prot_human, 'zebrafish': prot_zebrafish, 'celegans': prot_celegans, 'yeast': prot_yeast, 'athaliana': prot_athaliana}, f)
#
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/locations.pkl', 'wb') as f:
#     pickle.dump({'mouse': loc_mouse, 'rat': loc_rat, 'human': loc_human, 'zebrafish': loc_zebrafish, 'celegans': loc_celegans,
#                  'yeast': loc_yeast, 'athaliana': loc_athaliana}, f)
#
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/all_ylabels.pkl', 'wb') as f:
#     pickle.dump({'mouse': ymouse, 'rat': yrat, 'human': yhuman, 'zebrafish': yzebrafish, 'celegans': ycelegans,
#                  'yeast': yyeast, 'athaliana': yathaliana}, f)