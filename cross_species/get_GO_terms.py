import numpy as np
import pickle

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/unique_per_species.pkl', 'rb') as f:
    unique = pickle.load(f)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/mouse.pkl', 'rb') as f:
    ymouse, Xm, prot_mouse, loc_mouse = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/rat.pkl', 'rb') as f:
    yrat, Xrat, prot_rat, loc_rat = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/celegans.pkl', 'rb') as f:
    ycelegans, Xcelegans, prot_celegans, loc_celegans = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/yeast.pkl', 'rb') as f:
    yyeast, Xyeast, prot_yeast, loc_yeast = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/human.pkl', 'rb') as f:
    yhuman, Xhuman, prot_human, loc_human = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/zebrafish.pkl', 'rb') as f:
    yzebrafish, Xzebrafish, prot_zebrafish, loc_zebrafish = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/athaliana.pkl', 'rb') as f:
    yathaliana, Xathaliana, prot_athaliana, loc_athaliana = pickle.load(f)


with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/human_index_setsX.pkl', 'rb') as f:
    Xhuman_train, Xhuman_valid, Xhuman_test = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/human_index_setsY.pkl', 'rb') as f:
    yhuman_train, yhuman_valid, yhuman_test = pickle.load(f)

 

###################### for protein/term centric change folder
term = []
print('human')
print('number unique GO terms: %s' % len(unique['human']))
for key in loc_human.keys():
    if np.sum(yhuman_train[:, loc_human[key]]) >= 5:
        term.append(key)
print('number with >= 5 annotations in train: %s' % len(term))
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/term_mouse_train.pkl', 'wb') as f:
    pickle.dump(term, f)



def index_test(species, y, loc):
    term1 = []
    count = 0
    print(species)
    print('number unique GO terms: %s' % len(unique[species]))

    for key1 in loc.keys():
        if np.sum(y[:, loc[key1]]) >= 3:
            count = count + 1
            if key1 in term:
                term1.append(key1)
    print('at least 3 annotation %s' % count)
    print('number with >= 3 annotations and in selected mouse terms: %s' % len(term1))
    return term1


# beware! in folder index_protein_centric or index_term_centric
humanvalid_indexes = index_test('human', yhuman_valid, loc_human)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/term_human_valid.pkl', 'wb') as f:
    pickle.dump(humanvalid_indexes, f)
humantest_indexes = index_test('human', yhuman_test, loc_human)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/term_human_test.pkl', 'wb') as f:
    pickle.dump(humantest_indexes, f)
rat_indexes = index_test('rat', yrat, loc_rat)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/term_rat.pkl', 'wb') as f:
    pickle.dump(rat_indexes, f)
yeast_indexes = index_test('yeast', yyeast, loc_yeast)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/term_yeast.pkl', 'wb') as f:
    pickle.dump(yeast_indexes, f)
celegans_indexes = index_test('celegans', ycelegans, loc_celegans)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/term_celegans.pkl', 'wb') as f:
    pickle.dump(celegans_indexes, f)
mouse_indexes = index_test('mouse', ymouse, loc_mouse)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/term_mouse.pkl', 'wb') as f:
    pickle.dump(mouse_indexes, f)
zebrafish_indexes = index_test('zebrafish', yzebrafish, loc_zebrafish)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/term_zebrafish.pkl', 'wb') as f:
    pickle.dump(zebrafish_indexes, f)
athaliana_indexes = index_test('athaliana', yathaliana, loc_athaliana)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/term_athaliana.pkl', 'wb') as f:
    pickle.dump(athaliana_indexes, f)




###################### for protein/term centric change folder
termjoe = []
print('human')
print('number unique GO terms: %s' % len(unique['human']))
for key in loc_human.keys():
    if np.sum(yhuman_train[:, loc_human[key]]) >= 1:
        termjoe.append(key)
print('number with >= 1 annotations in train: %s' % len(termjoe))
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/protein_mouse_train.pkl', 'wb') as f:
    pickle.dump(termjoe, f)



def index_test(species, y, loc):
    term11 = []
    print(species)
    print('number unique GO terms: %s' % len(unique[species]))

    for key11 in loc.keys():
        if np.sum(y[:, loc[key11]]) >= 1:
            if key11 in termjoe:
                term11.append(key11)
    print('number with >= 1 annotations and in selected mouse terms: %s' % len(term11))
    return term11


# beware! in folder index_protein_centric or index_term_centric
humanvalid_indexes = index_test('human', yhuman_valid, loc_human)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/protein_human_valid.pkl', 'wb') as f:
    pickle.dump(humanvalid_indexes, f)
humantest_indexes = index_test('human', yhuman_test, loc_human)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/protein_human_test.pkl', 'wb') as f:
    pickle.dump(humantest_indexes, f)
rat_indexes = index_test('rat', yrat, loc_rat)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/protein_rat.pkl', 'wb') as f:
    pickle.dump(rat_indexes, f)
yeast_indexes = index_test('yeast', yyeast, loc_yeast)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/protein_yeast.pkl', 'wb') as f:
    pickle.dump(yeast_indexes, f)
celegans_indexes = index_test('celegans', ycelegans, loc_celegans)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/protein_celegans.pkl', 'wb') as f:
    pickle.dump(celegans_indexes, f)
mouse_indexes = index_test('mouse', ymouse, loc_mouse)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/protein_mouse.pkl', 'wb') as f:
    pickle.dump(mouse_indexes, f)
zebrafish_indexes = index_test('zebrafish', yzebrafish, loc_zebrafish)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/protein_zebrafish.pkl', 'wb') as f:
    pickle.dump(zebrafish_indexes, f)
athaliana_indexes = index_test('athaliana', yathaliana, loc_athaliana)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/protein_athaliana.pkl', 'wb') as f:
    pickle.dump(athaliana_indexes, f)












# # calculate information content values
# def IC_val(y, loc, species):
#     num = np.sum(y, axis=0)
#     tot = np.sum(y)
#
#     IC = {}
#     tot_IC = 0
#     for term in loc.keys():
#         IC[term] = -np.log(num[loc[term]] / tot)
#         tot_IC = tot_IC + -np.log(num[loc[term]] / tot)
#
#     with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_term_centric/' + species + '.pkl','rb') as f:
#         term_index = pickle.load(f)
#     with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/' + species + '.pkl','rb') as f:
#         protein_index = pickle.load(f)
#
#     tot_prot = 0
#     for i in protein_index:
#         tot_prot = tot_prot + IC[i]
#
#     tot_term = 0
#     for ii in term_index:
#         tot_term= tot_term + IC[ii]
#
#     return tot_IC, tot_prot/tot_IC, tot_term/tot_IC
#
# print(IC_val(yrat, loc_rat, 'rat'))
# print(IC_val(yhuman, loc_human, 'human'))
# print(IC_val(yzebrafish, loc_zebrafish, 'zebrafish'))
# print(IC_val(ycelegans, loc_celegans, 'celegans'))
# print(IC_val(yyeast, loc_yeast, 'yeast'))
# print(IC_val(yathaliana, loc_athaliana, 'athaliana'))
#
# #for the mouse sets
# def IC_val(y, loc, species):
#     num = np.sum(y, axis=0)
#     tot = np.sum(y)
#
#     with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_term_centric/' + species + '.pkl','rb') as f:
#         term_index = pickle.load(f)
#     with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/' + species + '.pkl','rb') as f:
#         protein_index = pickle.load(f)
#
#     IC = {}
#     tot_IC = 0
#     for term in loc.keys():
#         if num[loc[term]] != 0:
#             IC[term] = -np.log(num[loc[term]] / tot)
#             tot_IC = tot_IC + -np.log(num[loc[term]] / tot)
#
#     tot_prot = 0
#     for i in protein_index:
#         tot_prot = tot_prot + IC[i]
#
#     tot_term = 0
#     for ii in term_index:
#         tot_term= tot_term + IC[ii]
#
#     return tot_IC, tot_prot/tot_IC, tot_term/tot_IC
#
# print(IC_val(ymouse_valid, loc_mouse, 'mouse_valid'))
# print(IC_val(ymouse_test, loc_mouse, 'mouse_test'))