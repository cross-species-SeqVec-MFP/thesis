import numpy as np
import pickle

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GAF_files/Experimental_id/rat.pkl', 'rb') as f:
    id_rat = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GAF_files/Experimental_id/mouse.pkl', 'rb') as f:
    id_mouse = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GAF_files/Experimental_id/celegans.pkl', 'rb') as f:
    id_celegans = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GAF_files/Experimental_id/yeast.pkl', 'rb') as f:
    id_yeast = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GAF_files/Experimental_id/human.pkl', 'rb') as f:
    id_human = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GAF_files/Experimental_id/zebrafish.pkl', 'rb') as f:
    id_zebrafish = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GAF_files/Experimental_id/athaliana.pkl', 'rb') as f:
    id_athaliana = pickle.load(f)


def GO_terms_per_protein(dir_tab, protein_id):
    dict = {}
    with open(dir_tab) as f:
        for line in f:
            if line.split('\t')[0] in protein_id:
                terms1 = line.split('\t')[4]
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
                        with open(
                                "/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GO_terms/GO_ancestors/%s.txt" % term) as f:
                            for line1 in f:
                                line1 = line1.strip("\n")
                                line1 = line1.strip().split(" ")
                                if line1[0] == '*':
                                    GO_terms.append(line1[2])
                                else:
                                    GO_terms.append(line1[1])

                    dict[line.split('\t')[0]] = list(set(GO for GO in GO_terms))
    return dict


terms_rat = GO_terms_per_protein('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/'
                                 'Sequence_fasta_tab_files/rat_protein_GO.tab', id_rat)
terms_mouse = GO_terms_per_protein('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/'
                                   'Sequence_fasta_tab_files/mouse_protein_GO.tab', id_mouse)
terms_celegans = GO_terms_per_protein('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/'
                                      'Sequence_fasta_tab_files/c.elegans_protein_GO.tab', id_celegans)
terms_yeast = GO_terms_per_protein('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/'
                                   'Sequence_fasta_tab_files/yeast_protein_GO.tab', id_yeast)
terms_human = GO_terms_per_protein('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/'
                                   'Sequence_fasta_tab_files/human_protein_GO.tab', id_human)
terms_zebrafish = GO_terms_per_protein('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/'
                                       'Sequence_fasta_tab_files/zebrafish_protein_GO.tab', id_zebrafish)
terms_athaliana = GO_terms_per_protein('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/'
                                       'Sequence_fasta_tab_files/a.thaliana_protein_GO.tab', id_athaliana)

terms = [terms_rat, terms_mouse, terms_celegans, terms_yeast, terms_human, terms_zebrafish, terms_athaliana]
all_unique_GO = list(set(val2 for term in terms for val in term.values() for val2 in val))
remaining_terms = all_unique_GO

unique_mouse = list(set(val2 for val in terms_mouse.values() for val2 in val))
unique_rat = list(set(val2 for val in terms_rat.values() for val2 in val))
unique_celegans = list(set(val2 for val in terms_celegans.values() for val2 in val))
unique_yeast = list(set(val2 for val in terms_yeast.values() for val2 in val))
unique_human = list(set(val2 for val in terms_human.values() for val2 in val))
unique_zebrafish = list(set(val2 for val in terms_zebrafish.values() for val2 in val))
unique_athaliana = list(set(val2 for val in terms_athaliana.values() for val2 in val))

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/unique_per_species.pkl', 'wb') as f:
    pickle.dump({'all': all_unique_GO, 'mouse': unique_mouse, 'rat': unique_rat, 'celegans': unique_celegans, 'yeast': unique_yeast, 'human': unique_human, 'zebrafish': unique_zebrafish, 'athaliana': unique_athaliana}, f)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/embeddings/Xmouse_genmean', 'rb') as k:
    Xmouse = pickle.load(k)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/embeddings/Xrat_genmean', 'rb') as k:
    Xrat = pickle.load(k)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/embeddings/Xcelegans_genmean', 'rb') as k:
    Xcelegans = pickle.load(k)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/embeddings/Xyeast_genmean', 'rb') as k:
    Xyeast = pickle.load(k)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/embeddings/Xhuman_genmean', 'rb') as k:
    Xhuman = pickle.load(k)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/embeddings/Xzebrafish_genmean', 'rb') as k:
    Xzebrafish = pickle.load(k)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/embeddings/Xathaliana_genmean', 'rb') as k:
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
    Xfilt = np.zeros((len(dicti.keys()), 10240))
    prot_id = np.zeros((len(dicti.keys()), ), dtype=object)

    for x, id2 in enumerate(count_unique):
        prot_id[x] = id2
        y[x, :] = count_unique[id2]
        Xfilt[x, :] = X[id2].reshape(1024 * 10, order='F')
    return prot_id, y, Xfilt, location_term


prot_mouse, ymouse, Xm, location_mouse = count_GO(terms_mouse, unique_mouse, Xmouse)
prot_rat, yrat, Xr, location_rat = count_GO(terms_rat, unique_rat, Xrat)
prot_celegans, ycelegans, Xc, location_celegans = count_GO(terms_celegans, unique_celegans, Xcelegans)
prot_yeast, yyeast, Xy, location_yeast = count_GO(terms_yeast, unique_yeast, Xyeast)
prot_human, yhuman, Xh, location_human = count_GO(terms_human, unique_human, Xhuman)
prot_zebrafish, yzebrafish, Xz, location_zebrafish = count_GO(terms_zebrafish, unique_zebrafish, Xzebrafish)
prot_athaliana, yathaliana, Xa, location_athaliana = count_GO(terms_athaliana, unique_athaliana, Xathaliana)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/mouse.pkl', 'wb') as f:
    pickle.dump([ymouse, Xm, prot_mouse, location_mouse], f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/rat.pkl', 'wb') as f:
    pickle.dump([yrat, Xr, prot_rat, location_rat], f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/celegans.pkl', 'wb') as f:
    pickle.dump([ycelegans, Xc, prot_celegans, location_celegans], f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/yeast.pkl', 'wb') as f:
    pickle.dump([yyeast, Xy, prot_yeast, location_yeast], f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/human.pkl', 'wb') as f:
    pickle.dump([yhuman, Xh, prot_human, location_human], f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/zebrafish.pkl', 'wb') as f:
    pickle.dump([yzebrafish, Xz, prot_zebrafish, location_zebrafish], f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/athaliana.pkl', 'wb') as f:
    pickle.dump([yathaliana, Xa, prot_athaliana, location_athaliana], f)




with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/protein_names.pkl', 'rb') as f:
    data = pickle.load(f)







# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/unique_GO_part_1.txt', 'r') as f:
#     for line_id in f:
#         line_id = line_id.strip("\n")
#         if line_id in all_unique_GO:
#             remaining_terms.remove(line_id)
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/unique_GO_part_2.txt', 'r') as f:
#     for line_id1 in f:
#         line_id1 = line_id1.strip("\n")
#         if line_id1 in all_unique_GO:
#             remaining_terms.remove(line_id1)
#
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/unique_GO_part_3.txt', 'w') as f:
#     for item in remaining_terms:
#         f.write("%s\n" % item)
#     f.write('complete')







# #####################################################
# # getting the GO terms to test
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/mouse_index.pkl', 'rb') as f:
#     mouse_train_ind, mouse_valid_ind = pickle.load(f)
#
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/mouse.pkl', 'rb') as f:
#     ymouse, Xmouse = pickle.load(f)
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/rat.pkl', 'rb') as f:
#     yrat, Xrat = pickle.load(f)
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/celegans.pkl', 'rb') as f:
#     ycelegans, Xcelegans = pickle.load(f)
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/yeast.pkl', 'rb') as f:
#     yyeast, Xyeast = pickle.load(f)
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/human.pkl', 'rb') as f:
#     yhuman, Xhuman = pickle.load(f)
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/zebrafish.pkl', 'rb') as f:
#     yzebrafish, Xzebrafish = pickle.load(f)
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/athaliana.pkl', 'rb') as f:
#     yathaliana, Xathaliana = pickle.load(f)
#
# Xmouse_train = Xmouse[mouse_train_ind, :]
# Xmouse_valid = Xmouse[mouse_valid_ind, :]
# ymouse_train = ymouse[mouse_train_ind, :]
# ymouse_valid = ymouse[mouse_valid_ind, :]
#
# count_filt_train = np.sum(ymouse_train, axis=0)
# count_filt_valid = np.sum(ymouse_valid, axis=0)
#
# index_train = np.array(np.where(count_filt_train >= 40))
# index_valid = np.array(np.where(count_filt_valid >= 5))
# overlap = np.isin(index_train, index_valid)
# row, col = np.nonzero(overlap)
# GO_terms_mouse = index_train.reshape(-1, 1)[col]
# testable_terms = np.array(unique_mouse)[GO_terms_mouse]
#
# def test_terms(unique):
#     row, col = np.isin(unique, testable_terms)
#
#
# count = np.zeros((len(unique_mouse),))
# for id in terms_mouse.keys():
#     count = np.add(count, count_unique_mouse[id])
#
# hist = np.histogram(count, bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, len(terms_mouse.keys())])
# index = np.where(count >= 45)
# unique_mouse = np.array(unique_mouse)
# GO_above401 = unique_mouse[index]

#
# overlap_rat = list(GO for GO in unique_rat if GO in unique_mouse)
# overlap_yeast = list(GO for GO in unique_yeast if GO in unique_mouse)
# overlap_celegans = list(GO for GO in unique_celegans if GO in unique_mouse)
# overlap_human = list(GO for GO in unique_human if GO in unique_mouse)
# overlap_zebrafish = list(GO for GO in unique_zebrafish if GO in unique_mouse)
# overlap_athaliana = list(GO for GO in unique_athaliana if GO in unique_mouse)
#
#
# overlap_rat = list(GO for GO in unique_rat if GO in term_mouse)
# overlap_yeast = list(GO for GO in unique_yeast if GO in GO_above40)
# overlap_celegans = list(GO for GO in unique_celegans if GO in GO_above40)
# overlap_human = list(GO for GO in unique_human if GO in GO_above40)
# overlap_zebrafish = list(GO for GO in unique_zebrafish if GO in GO_above40)
# overlap_athaliana = list(GO for GO in unique_athaliana if GO in GO_above40)


#
# def histo_count(overlap, count_unique, dicti):
#     count = np.zeros((len(overlap),))
#     for id in dicti.keys():
#         count = np.add(count, count_unique[id])
#     hist = np.histogram(count, bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 20000])
#     return(hist)
#
#
# hist_rat = histo_count(overlap_rat, count_unique_rat, terms_rat)
# hist_celegans = histo_count(overlap_celegans, count_unique_celegans, terms_celegans)
# hist_yeast = histo_count(overlap_yeast, count_unique_yeast, terms_yeast)
# hist_human = histo_count(overlap_human, count_unique_human, terms_human)
# hist_zebrafish = histo_count(overlap_zebrafish, count_unique_zebrafish, terms_zebrafish)
# hist_athaliana = histo_count(overlap_athaliana, count_unique_athaliana, terms_athaliana)









# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/unique_GO_part_1.txt', 'w') as f:
#     for item in all_unique_GO[0:2733]:
#         f.write("%s\n" % item)
#     f.write('complete')
#
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/unique_GO_part_2.txt', 'w') as f:
#     for item in all_unique_GO[2733:len(all_unique_GO)]:
#         f.write("%s\n" % item)
#     f.write('complete')

# while IFS= read -r line; do
#     /home/nfs/ivandenbent/bubble/bin/wr_hier.py $line --concise -o /tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GO_terms/GO_descendants/$line.txt
# done < /tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/unique_GO_part_2.txt



