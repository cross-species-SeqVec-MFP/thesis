import numpy as np
import pickle
import sys
# this code gives all the GO annotations per protein, including annotations from ancestor GO terms to
# GO annotations.

directory = sys.argv[1]
directory1 = sys.argv[2]
directory2 = sys.argv[3]
directory3 = sys.argv[4]


with open('%s/rat.pkl' % directory, 'rb') as f:
    id_rat = pickle.load(f)
with open('%s/mouse.pkl' % directory, 'rb') as f:
    id_mouse = pickle.load(f)
with open('%s/celegans.pkl' % directory, 'rb') as f:
    id_celegans = pickle.load(f)
with open('%s/yeast.pkl' % directory, 'rb') as f:
    id_yeast = pickle.load(f)
with open('%s/human.pkl' % directory, 'rb') as f:
    id_human = pickle.load(f)
with open('%s/zebrafish.pkl' % directory, 'rb') as f:
    id_zebrafish = pickle.load(f)
with open('%s/athaliana.pkl' % directory, 'rb') as f:
    id_athaliana = pickle.load(f)


def GO_terms_per_protein(dir_tab, protein_id):
    dict = {}
    with open('%s/%s' % (directory1, dir_tab)) as f:
        for line in f:
            if line.split('\t')[0] in protein_id:
                terms1 = line.split('\t')[1]
                GO_annotations = []
                inRecordingMode = False
                for ind, x in enumerate(terms1):
                    if not inRecordingMode:
                        if x == '[' and terms1[ind + 1] == 'G':
                            inRecordingMode = True
                            string = ''
                    elif x == ']':
                        inRecordingMode = False
                        GO_annotations.append(string)
                    if inRecordingMode and x != '[':
                        string = string + x

                if GO_annotations:

                    GO_terms = []
                    for i, term in enumerate(GO_annotations):
                        with open("%s/GO_ancestors/%s.txt" % (directory, term)) as f:
                            for line1 in f:
                                line1 = line1.strip("\n")
                                line1 = line1.strip().split(" ")
                                if line1[0] == '*':
                                    GO_terms.append(line1[2])
                                else:
                                    GO_terms.append(line1[1])

                    dict[line.split('\t')[0]] = list(set(GO for GO in GO_terms))

    return dict


terms_rat = GO_terms_per_protein('rat_protein_GO.tab', id_rat)
terms_mouse = GO_terms_per_protein('mouse_protein_GO.tab', id_mouse)
terms_celegans = GO_terms_per_protein('c.elegans_protein_GO.tab', id_celegans)
terms_yeast = GO_terms_per_protein('yeast_protein_GO.tab', id_yeast)
terms_human = GO_terms_per_protein('human_protein_GO.tab', id_human)
terms_zebrafish = GO_terms_per_protein('zebrafish_protein_GO.tab', id_zebrafish)
terms_athaliana = GO_terms_per_protein('a.thaliana_protein_GO.tab', id_athaliana)

terms = [terms_rat, terms_mouse, terms_celegans, terms_yeast, terms_human, terms_zebrafish, terms_athaliana]
all_unique_GO = list(set(val2 for term in terms for val in term.values() for val2 in val))

unique_mouse = list(set(val2 for val in terms_mouse.values() for val2 in val))
unique_rat = list(set(val2 for val in terms_rat.values() for val2 in val))
unique_celegans = list(set(val2 for val in terms_celegans.values() for val2 in val))
unique_yeast = list(set(val2 for val in terms_yeast.values() for val2 in val))
unique_human = list(set(val2 for val in terms_human.values() for val2 in val))
unique_zebrafish = list(set(val2 for val in terms_zebrafish.values() for val2 in val))
unique_athaliana = list(set(val2 for val in terms_athaliana.values() for val2 in val))



with open('%s/unique_per_species.pkl' % directory3, 'wb') as f:
    pickle.dump({'all': all_unique_GO, 'mouse': unique_mouse, 'rat': unique_rat, 'celegans': unique_celegans, 'yeast': unique_yeast, 'human': unique_human, 'zebrafish': unique_zebrafish, 'athaliana': unique_athaliana}, f)

with open('%s/Xmouse' % directory1, 'rb') as k:
    Xmouse = pickle.load(k)
with open('%s/Xrat' % directory1, 'rb') as k:
    Xrat = pickle.load(k)
with open('%s/Xc.elegans' % directory1, 'rb') as k:
    Xcelegans = pickle.load(k)
with open('%s/Xyeast' % directory1, 'rb') as k:
    Xyeast = pickle.load(k)
with open('%s/Xhuman' % directory1, 'rb') as k:
    Xhuman = pickle.load(k)
with open('%s/Xzebrafish' % directory1, 'rb') as k:
    Xzebrafish = pickle.load(k)
with open('%s/Xa.thaliana' % directory1, 'rb') as k:
    Xathaliana = pickle.load(k)


def count_GO(dicti, GO_term, X):
    zero = np.zeros((len(dicti.keys()), len(GO_term)))
    count_unique = dict(zip(dicti.keys(), zero))
    location_term = {}
    for i, term in enumerate(GO_term):
        for id1 in dicti.keys():
            for val in dicti[id1]:
                if val == term:
                    count_unique[id1][i] = 1

        location_term[term] = i

    y = np.zeros((len(dicti.keys()), len(GO_term)))
    Xfilt = np.zeros((len(dicti.keys()), 1024))
    prot_id = np.zeros((len(dicti.keys()), ), dtype=object)

    for x, id2 in enumerate(count_unique):
        prot_id[x] = id2
        y[x, :] = count_unique[id2]
        Xfilt[x, :] = X[id2]
    return prot_id, y, Xfilt, location_term


prot_mouse, ymouse, Xm, location_mouse = count_GO(terms_mouse, unique_mouse, Xmouse)
prot_rat, yrat, Xr, location_rat = count_GO(terms_rat, unique_rat, Xrat)
prot_celegans, ycelegans, Xc, location_celegans = count_GO(terms_celegans, unique_celegans, Xcelegans)
prot_yeast, yyeast, Xy, location_yeast = count_GO(terms_yeast, unique_yeast, Xyeast)
prot_human, yhuman, Xh, location_human = count_GO(terms_human, unique_human, Xhuman)
prot_zebrafish, yzebrafish, Xz, location_zebrafish = count_GO(terms_zebrafish, unique_zebrafish, Xzebrafish)
prot_athaliana, yathaliana, Xa, location_athaliana = count_GO(terms_athaliana, unique_athaliana, Xathaliana)

with open('%s/mouse.pkl' % directory2, 'wb') as f:
    pickle.dump([ymouse, Xm, prot_mouse, location_mouse], f)
with open('%s/rat.pkl' % directory2, 'wb') as f:
    pickle.dump([yrat, Xr, prot_rat, location_rat], f)
with open('%s/celegans.pkl' % directory2, 'wb') as f:
    pickle.dump([ycelegans, Xc, prot_celegans, location_celegans], f)
with open('%s/yeast.pkl' % directory2, 'wb') as f:
    pickle.dump([yyeast, Xy, prot_yeast, location_yeast], f)
with open('%s/human.pkl' % directory2, 'wb') as f:
    pickle.dump([yhuman, Xh, prot_human, location_human], f)
with open('%s/zebrafish.pkl' % directory2, 'wb') as f:
    pickle.dump([yzebrafish, Xz, prot_zebrafish, location_zebrafish], f)
with open('%s/athaliana.pkl' % directory2, 'wb') as f:
    pickle.dump([yathaliana, Xa, prot_athaliana, location_athaliana], f)
