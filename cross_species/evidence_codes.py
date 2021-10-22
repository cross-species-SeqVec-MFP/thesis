from numpy import load
import numpy as np
import pickle
import sys
# This codes ensures that only the proteins with experimentally derived GO annotations
# are included in the experiments

embeddingsPath = sys.argv[1]
evidenceCodesPath = sys.argv[2]
directory = sys.argv[3]
directory1 = sys.argv[4]

with open(embeddingsPath + '/Xrat', 'rb') as f:
    Xrat = pickle.load(f)
with open(embeddingsPath + '/Xmouse', 'rb') as g:
    Xmouse = pickle.load(g)
with open(embeddingsPath + '/Xc.elegans', 'rb') as h:
    Xcelegans = pickle.load(h)
with open(embeddingsPath + '/Xyeast', 'rb') as k:
    Xyeast = pickle.load(k)
with open(embeddingsPath + '/Xhuman', 'rb') as k:
    Xhuman = pickle.load(k)
with open(embeddingsPath + '/Xa.thaliana', 'rb') as k:
    Xathaliana = pickle.load(k)
with open(embeddingsPath + '/Xzebrafish', 'rb') as k:
    Xzebrafish = pickle.load(k)

id_human = Xhuman.keys()
id_athaliana = Xathaliana.keys()
id_zebrafish = Xzebrafish.keys()
id_rat = Xrat.keys()
id_mouse = Xmouse.keys()
id_celegans = Xcelegans.keys()
id_yeast = Xyeast.keys()

def experimental_id(id_species):

    evidence_codes = ['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP',
                      'HTP', 'HDA', 'HMP', 'HGI', 'HEP',
                      'IBA', 'IBD', 'IKR', 'IRD', 'IC', 'TAS']
    evidence_id = []
    with open(evidenceCodesPath + "/evidence_codes.gaf") as f:
        for line in f:
            line = line.strip("\n")
            line = line.strip().split("\t")
            if line[1] in id_species:
                if line[6] in evidence_codes:
                    evidence_id.append(line[1])
        evidence_id = np.array(evidence_id)
        evi_id = np.unique(evidence_id)
    return evi_id


evi_id_rat = experimental_id(id_rat)
evi_id_mouse = experimental_id(id_mouse)
evi_id_celegans = experimental_id(id_celegans)
evi_id_yeast = experimental_id(id_yeast)
evi_id_human = experimental_id(id_human)
evi_id_athaliana = experimental_id(id_athaliana)
evi_id_zebrafish = experimental_id(id_zebrafish)

with open(evidenceCodesPath + '/rat.pkl', 'wb') as f:
    pickle.dump(evi_id_rat, f)
with open(evidenceCodesPath + '/mouse.pkl', 'wb') as f:
    pickle.dump(evi_id_mouse, f)
with open(evidenceCodesPath + '/celegans.pkl', 'wb') as f:
    pickle.dump(evi_id_celegans, f)
with open(evidenceCodesPath + '/yeast.pkl', 'wb') as f:
    pickle.dump(evi_id_yeast, f)
with open(evidenceCodesPath + '/human.pkl', 'wb') as f:
    pickle.dump(evi_id_human, f)
with open(evidenceCodesPath + '/athaliana.pkl', 'wb') as f:
    pickle.dump(evi_id_athaliana, f)
with open(evidenceCodesPath + '/zebrafish.pkl', 'wb') as f:
    pickle.dump(evi_id_zebrafish, f)


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

                    dict[line.split('\t')[0]] = list(set(GO for GO in GO_annotations))
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

# make a list with all the GO terms that are in the cross-species test sets

with open('%s/list_terms.txt' % directory, 'w') as f:
    for item in all_unique_GO:
        f.write("%s\n" % item)
    f.write('complete')
