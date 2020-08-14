import numpy as np
import pickle
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from statistics import mean
import matplotlib.pyplot as plt
from datetime import datetime

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GO_terms/GO_depth_all_mouse_proteins.pkl', 'rb') as fw:
    depth_GO = pickle.load(fw)

def get_terms(species):
    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/validation_results/predictions_term_centric/%s.pkl' % species, 'rb') as fw:
        data = pickle.load(fw)

    rocauc = roc_auc_score(data['Yval'], data['Ypost'], average=None)

    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/BLAST/annotation_result/term_centric/%s.pkl' % species, 'rb') as fw:
        data_blast = pickle.load(fw)

    rocauc_blast = roc_auc_score(data_blast['Yval'], data_blast['Ypost'], average=None)

    df_species = []
    df_function = []
    df_rocauc = []
    df_speciesb = []
    df_functionb = []
    df_rocaucb = []
    avg_rocauc = {}
    avg_rocauc_numkids = {}

    kids_terms = {}
    for term in data['loc_term'].keys():
        if int(depth_GO[term][0]) == 2:
            with open("/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GO_terms/GO_descendants/%s.txt" % term) as f:
                lines = f.readlines()
            info = np.zeros(2, dtype=object)
            for i, x in enumerate(lines[0].strip("\n").split(" ")):
                if x.startswith('D'):
                    function = ""
                    for ii in range(i + 1, len(lines[0].strip("\n").split(" "))):
                        function = function + lines[0].strip("\n").split(" ")[ii] + " "
                    info[0] = function
                    break

            kids = []
            for i in np.arange(1, len(lines), 1):
                kids.append(lines[i].strip("\n").split(" ")[1])
            info[1] = kids
            kids_terms[term] = info


            avg = []
            avg_blast = []
            avg.append(rocauc[data['loc_term'][term]])
            avg_blast.append(rocauc_blast[data['loc_term'][term]])
            for kid in kids_terms[term][1]:
                if kid in data['loc_term'].keys():
                    avg.append(rocauc[data['loc_term'][kid]])
                    avg_blast.append(rocauc_blast[data['loc_term'][kid]])
            if len(avg) >= 5:
                avg_rocauc[term] = mean(avg)
                avg_rocauc_numkids[term] = len(avg)

                if kids_terms[term][0] != 'organic cyclic compound binding ':
                    df_rocauc.append(mean(avg))
                    df_species.append(species)
                    df_function.append(kids_terms[term][0])
                    df_rocaucb.append(mean(avg_blast))
                    df_speciesb.append(species)
                    df_functionb.append(kids_terms[term][0])

    return avg_rocauc, avg_rocauc_numkids, df_rocauc, df_function, df_species, df_rocaucb, df_functionb, df_speciesb


avg_rocauc_mouse, avg_rocauc_numkids_mouse, df_r_mouse, df_f_mouse, df_s_mouse, df_rb_mouse, df_fb_mouse, df_sb_mouse = get_terms('mouse_test')
avg_rocauc_rat, avg_rocauc_numkids_rat, df_r_rat, df_f_rat, df_s_rat, df_rb_rat, df_fb_rat, df_sb_rat = get_terms('rat')
avg_rocauc_human, avg_rocauc_numkids_human, df_r_human, df_f_human, df_s_human, df_rb_human, df_fb_human, df_sb_human = get_terms('human')
avg_rocauc_zebrafish, avg_rocauc_numkids_zebrafish, df_r_zebrafish, df_f_zebrafish, df_s_zebrafish, df_rb_zebrafish, df_fb_zebrafish, df_sb_zebrafish = get_terms('zebrafish')
avg_rocauc_celegans, avg_rocauc_numkids_celegans, df_r_celegans, df_f_celegans, df_s_celegans, df_rb_celegans, df_fb_celegans, df_sb_celegans = get_terms('celegans')
avg_rocauc_yeast, avg_rocauc_numkids_yeast, df_r_yeast, df_f_yeast, df_s_yeast, df_rb_yeast, df_fb_yeast, df_sb_yeast = get_terms('yeast')
avg_rocauc_athaliana, avg_rocauc_numkids_athaliana, df_r_athaliana, df_f_athaliana, df_s_athaliana, df_rb_athaliana, df_fb_athaliana, df_sb_athaliana = get_terms('athaliana')

all_df_rb = df_rb_mouse + df_rb_rat + df_rb_human + df_rb_zebrafish + df_rb_celegans + df_rb_yeast + df_rb_athaliana
all_df_f = df_f_mouse + df_f_rat + df_f_human + df_f_zebrafish + df_f_celegans + df_f_yeast + df_f_athaliana
all_df_r = df_r_mouse + df_r_rat + df_r_human + df_r_zebrafish + df_r_celegans + df_r_yeast + df_r_athaliana
all_df_s = df_s_mouse + df_s_rat + df_s_human + df_s_zebrafish + df_s_celegans + df_s_yeast + df_s_athaliana
unique_f = list(set(func for func in all_df_f))

#######################3 terms overlapping in all species
elems_in_all = set(df_f_mouse) & set(df_f_rat) & set(df_f_human) & set(df_f_zebrafish) & set(df_f_celegans) & set(df_f_yeast) & set(df_f_athaliana)

df_in_all = pd.DataFrame({'Species': all_df_s, 'GO_term': all_df_f, 'Rocauc score': all_df_r})

# to_remove = unique_f
# for ele in elems_in_all:
#     to_remove.remove(ele)
#
# for ele in to_remove:
#     df_in_all = df_in_all[df_in_all.GO_term != ele]


#######################3 terms overlapping in at least 6 species
elems_in_6 = []
for ele1 in unique_f:
    summie = np.zeros(7)
    summie[0] = int(ele1 in df_f_mouse)
    summie[1] = int(ele1 in df_f_rat)
    summie[2] = int(ele1 in df_f_human)
    summie[3] = int(ele1 in df_f_zebrafish)
    summie[4] = int(ele1 in df_f_celegans)
    summie[5] = int(ele1 in df_f_yeast)
    summie[6] = int(ele1 in df_f_athaliana)
    if np.sum(summie) >= 6:
        elems_in_6.append(ele1)

df_in_all = pd.DataFrame({'Species': all_df_s, 'GO_term': all_df_f, 'Rocauc score': all_df_r})

to_remove = unique_f
for ele in elems_in_6:
    to_remove.remove(ele)

for ele in to_remove:
    df_in_all = df_in_all[df_in_all.GO_term != ele]

ordah = ['signaling receptor activity ', 'transmembrane transporter activity ',
         'cofactor binding ', 'lyase activity ',
         'receptor regulator activity ', 'catalytic activity, acting on RNA ',
         'transferase activity ', 'ion binding ',
         'catalytic activity, acting on a protein ', 'lipid binding ',
         'enzyme regulator activity ', 'DNA-binding transcription factor activity ',
         'protein binding ', 'oxidoreductase activity ',
         'hydrolase activity ', 'protein-containing complex binding ',
         'heterocyclic compound binding ',
         'small molecule binding ', 'catalytic activity, acting on DNA ',
         'carbohydrate derivative binding ']

spe = ['mouse_test', 'rat', 'human', 'zebrafish', 'celegans', 'yeast', 'athaliana']

result = df_in_all[df_in_all.GO_term.isin(ordah)]
result = result.pivot(index='GO_term', columns='Species', values='Rocauc score')
result = result.reindex(index=ordah, columns=spe)


result.index = pd.CategoricalIndex(result.index, categories= ordah)
result.sort_index(level=0, inplace=True)

plt.figure(figsize=(8, 6))
sns.heatmap(result, annot=True, fmt=".3f", cmap='Oranges')
now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.savefig('heatmap' + current_time + '.pdf')
plt.close()












now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.figure(figsize=[9, 11])
sns.set(style="whitegrid")
sns.swarmplot(y="GO_term", x="Rocauc score", hue='Species', data=df_in_all, size=6, order=ordah)
plt.title('across species function performance')
sns.boxplot(y="GO_term", x="Rocauc score", data=df_in_all, order =ordah ,medianprops = {'color': 'gray', 'linewidth': 1}, capprops = {'color': 'gray', 'linewidth': 1}, whiskerprops = {'color': 'gray', 'linewidth': 1}, boxprops={'facecolor': 'None', 'edgecolor': 'gray', 'linewidth': 1}, showfliers=False)
plt.xlim([0.48, 1.02])
plt.tight_layout()
plt.savefig('beeswarm_type_GOterm_species' + current_time + '.png')
plt.savefig('beeswarm_type_GOterm_species' + current_time + '.pdf')
plt.close()


############################################################3
df_in_all = pd.DataFrame({'Species': all_df_s, 'GO_term': all_df_f, 'Rocauc score': all_df_rb})

to_remove = unique_f
for ele in elems_in_6:
    to_remove.remove(ele)

for ele in to_remove:
    df_in_all = df_in_all[df_in_all.GO_term != ele]


now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.figure(figsize=[9, 11])
sns.set(style="whitegrid")
sns.swarmplot(y="GO_term", x="Rocauc score", hue='Species', data=df_in_all, size=6, order=ordah)
plt.title('across species function performance')
sns.boxplot(y="GO_term", x="Rocauc score", data=df_in_all, order =ordah ,medianprops = {'color': 'gray', 'linewidth': 1}, capprops = {'color': 'gray', 'linewidth': 1}, whiskerprops = {'color': 'gray', 'linewidth': 1}, boxprops={'facecolor': 'None', 'edgecolor': 'gray', 'linewidth': 1}, showfliers=False)
plt.xlim([0.48, 1.02])
plt.tight_layout()
plt.savefig('beeswarm_type_GOterm_species_blast' + current_time + '.png')
plt.savefig('beeswarm_type_GOterm_species_blast' + current_time + '.pdf')
plt.close()


