import numpy as np
import pickle
import pandas as pd
import sys
# code identifies which GO terms are evaluated for protein-centric and term-centric evaluation.
# code to calculate the Resnik information content (IC) is also included.

directory = sys.argv[1]

directory1 = sys.argv[2]

with open('%s/unique_per_species.pkl' % directory1, 'rb') as f:
    unique = pickle.load(f)

with open('%s/mouse.pkl' % directory, 'rb') as f:
    ymouse, Xm, prot_mouse, loc_mouse = pickle.load(f)
with open('%s/mouse_index_setsX.pkl' % directory, 'rb') as f:
    Xmouse_train, Xmouse_valid, Xmouse_test = pickle.load(f)
with open('%s/mouse_index_setsY.pkl' % directory, 'rb') as f:
    ymouse_train, ymouse_valid, ymouse_test = pickle.load(f)

with open('%s/rat.pkl' % directory, 'rb') as f:
    yrat, Xrat, prot_rat, loc_rat = pickle.load(f)
with open('%s/celegans.pkl' % directory, 'rb') as f:
    ycelegans, Xcelegans, prot_celegans, loc_celegans = pickle.load(f)
with open('%s/yeast.pkl' % directory, 'rb') as f:
    yyeast, Xyeast, prot_yeast, loc_yeast = pickle.load(f)
with open('%s/human.pkl' % directory, 'rb') as f:
    yhuman, Xhuman, prot_human, loc_human = pickle.load(f)
with open('%s/zebrafish.pkl' % directory, 'rb') as f:
    yzebrafish, Xzebrafish, prot_zebrafish, loc_zebrafish = pickle.load(f)
with open('%s/athaliana.pkl' % directory, 'rb') as f:
    yathaliana, Xathaliana, prot_athaliana, loc_athaliana = pickle.load(f)


# ###################### for term centric
print('term_centric')
term_mouse = []
print('mouse')
print('number unique GO terms: %s' % len(unique['mouse']))
for key in loc_mouse.keys():
    if np.sum(ymouse_train[:, loc_mouse[key]]) >= 5:
        term_mouse.append(key)
print('number with >= 5 annotations in train: %s' % len(term_mouse))
with open('%s/index_term_centric/mouse.pkl' % directory1, 'wb') as f:
    pickle.dump(term_mouse, f)



def index_test(species, y, loc):
    term1 = []
    count = 0
    joejoe = []
    print(species)
    print('number unique GO terms: %s' % len(unique[species]))
    print('Shape')
    print(y.shape)

    for key1 in loc.keys():
        if np.sum(y[:, loc[key1]]) >= 3:
            count = count + 1
            joejoe.append(key1)
            if key1 in term_mouse:
                term1.append(key1)
    print('number with >= 3 annotations and in selected mouse terms: %s' % len(term1))
    return term1

mousevalid_indexes = index_test('mouse', ymouse_valid, loc_mouse)
with open('%s/index_term_centric/mouse_valid.pkl' % directory1, 'wb') as f:
    pickle.dump(mousevalid_indexes, f)
mousetest_indexes = index_test('mouse', ymouse_test, loc_mouse)
with open('%s/index_term_centric/mouse_test.pkl' % directory1, 'wb') as f:
    pickle.dump(mousetest_indexes, f)
rat_indexes = index_test('rat', yrat, loc_rat)
with open('%s/index_term_centric/rat.pkl' % directory1, 'wb') as f:
    pickle.dump(rat_indexes, f)
yeast_indexes = index_test('yeast', yyeast, loc_yeast)
with open('%s/index_term_centric/yeast.pkl' % directory1, 'wb') as f:
    pickle.dump(yeast_indexes, f)
celegans_indexes = index_test('celegans', ycelegans, loc_celegans)
with open('%s/index_term_centric/celegans.pkl' % directory1, 'wb') as f:
    pickle.dump(celegans_indexes, f)
human_indexes = index_test('human', yhuman, loc_human)
with open('%s/index_term_centric/human.pkl' % directory1, 'wb') as f:
    pickle.dump(human_indexes, f)
zebrafish_indexes = index_test('zebrafish', yzebrafish, loc_zebrafish)
with open('%s/index_term_centric/zebrafish.pkl' % directory1, 'wb') as f:
    pickle.dump(zebrafish_indexes, f)
athaliana_indexes = index_test('athaliana', yathaliana, loc_athaliana)
with open('%s/index_term_centric/athaliana.pkl' % directory1, 'wb') as f:
    pickle.dump(athaliana_indexes, f)


# ###################### for protein centric
print('protein_centric')
term_mouse = []
print('mouse')
print('number unique GO terms: %s' % len(unique['mouse']))
for key in loc_mouse.keys():
    if np.sum(ymouse_train[:, loc_mouse[key]]) >= 1:
        term_mouse.append(key)
print('number with >= 1 annotations in train: %s' % len(term_mouse))
with open('%s/index_protein_centric/mouse.pkl' % directory1, 'wb') as f:
    pickle.dump(term_mouse, f)



def index_test(species, y, loc):
    term1 = []
    count = 0
    joejoe = []
    print(species)
    print('number unique GO terms: %s' % len(unique[species]))

    for key1 in loc.keys():
        if np.sum(y[:, loc[key1]]) >= 1:
            count = count + 1
            joejoe.append(key1)
            if key1 in term_mouse:
                term1.append(key1)
    print('number with >= 1 annotations and in selected mouse terms: %s' % len(term1))
    return term1


mousevalid_indexes = index_test('mouse', ymouse_valid, loc_mouse)
with open('%s/index_protein_centric/mouse_valid.pkl' % directory1, 'wb') as f:
    pickle.dump(mousevalid_indexes, f)
mousetest_indexes = index_test('mouse', ymouse_test, loc_mouse)
with open('%s/index_protein_centric/mouse_test.pkl' % directory1, 'wb') as f:
    pickle.dump(mousetest_indexes, f)
rat_indexes = index_test('rat', yrat, loc_rat)
with open('%s/index_protein_centric/rat.pkl' % directory1, 'wb') as f:
    pickle.dump(rat_indexes, f)
yeast_indexes = index_test('yeast', yyeast, loc_yeast)
with open('%s/index_protein_centric/yeast.pkl' % directory1, 'wb') as f:
    pickle.dump(yeast_indexes, f)
celegans_indexes = index_test('celegans', ycelegans, loc_celegans)
with open('%s/index_protein_centric/celegans.pkl' % directory1, 'wb') as f:
    pickle.dump(celegans_indexes, f)
human_indexes = index_test('human', yhuman, loc_human)
with open('%s/index_protein_centric/human.pkl' % directory1, 'wb') as f:
    pickle.dump(human_indexes, f)
zebrafish_indexes = index_test('zebrafish', yzebrafish, loc_zebrafish)
with open('%s/index_protein_centric/zebrafish.pkl' % directory1, 'wb') as f:
    pickle.dump(zebrafish_indexes, f)
athaliana_indexes = index_test('athaliana', yathaliana, loc_athaliana)
with open('%s/index_protein_centric/athaliana.pkl' % directory1, 'wb') as f:
    pickle.dump(athaliana_indexes, f)
