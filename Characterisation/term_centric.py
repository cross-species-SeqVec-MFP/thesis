import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from statistics import mean
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit
import sys

modelPath = sys.argv[1]


with open(modelPath + '/performance.pkl', 'rb') as f:
    dd = pickle.load(f)




#############################################
# statistical testing

ave_term_1024 = dd['tc']

with open('./GO_levels_all_GOterms.pkl', 'rb') as fw:
    depth_GO = pickle.load(fw)

levels = np.zeros((len(terms),))
depth_3_terms = []
depth_2_terms = []
depth_1_terms = []
for i, x in enumerate(terms):
    if int(depth_GO[x][0]) > 9:
        levels[i] = 9
    else:
        levels[i] = int(depth_GO[x][0])
        if int(depth_GO[x][0]) == 3:
            depth_3_terms.append(x)
        if int(depth_GO[x][0]) == 2:
            depth_2_terms.append(x)
        if int(depth_GO[x][0]) == 1:
            depth_1_terms.append(x)

#########################
# grouping terms
kids_terms = {}
for term in depth_2_terms:
    with open("./GO_descendants_%s.txt" % term) as f:
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

avg_rocauc = {}
avg_rocauc_numkids = {}
df_depth = []
df_function = []
df_rocauc = []
df_change = []
for term1 in kids_terms.keys():
    avg = []
    avg.append(ave_term_1024[np.where(np.array(terms) == term1)[0][0]])
    if kids_terms[term1][1]:
        for kid in kids_terms[term1][1]:
            if kid in terms:
                avg.append(ave_term_1024[np.where(np.array(terms) == kid)[0][0]])
    if len(avg) >= 5:
        avg_rocauc[term1] = mean(avg)
        avg_rocauc_numkids[term1] = len(avg)

        for kid in kids_terms[term1][1]:
            if kid in terms:
                if kids_terms[term1][0] != 'heterocyclic compound binding':
                    df_rocauc.append(ave_term_1024[np.where(np.array(terms) == kid)[0][0]])
                    df_depth.append(int(depth_GO[kid][0]))
                    df_function.append(kids_terms[term1][0])
                    df_change.append(improved[np.where(np.array(terms) == kid)[0][0]] - ave_term_1024[
                        np.where(np.array(terms) == kid)[0][0]])



df = pd.DataFrame({'GO term': df_function, 'Rocauc score': df_rocauc, 'GO term depth': df_depth})
ordah = ['signaling receptor activity ', 'transmembrane transporter activity ',
         'cofactor binding ', 'lyase activity ',
         'receptor regulator activity ', 'catalytic activity, acting on RNA ',
         'transferase activity ', 'ion binding ',
         'catalytic activity, acting on a protein ', 'lipid binding ',
         'enzyme regulator activity ', 'DNA-binding transcription factor activity ',
         'protein binding ', 'oxidoreductase activity ',
         'hydrolase activity ', 'protein-containing complex binding ',
         'organic cyclic compound binding ',
         'small molecule binding ', 'catalytic activity, acting on DNA ',
         'carbohydrate derivative binding ']

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.figure(figsize=[9, 10])
sns.set(style="whitegrid")
sns.swarmplot(y="GO term", x="Rocauc score", hue='GO term depth', zorder=1, data=df, size=3, order=ordah)
sns.boxplot(y="GO term", x="Rocauc score", data=df, order=ordah, boxprops={'facecolor': 'None', "zorder": 10},
            zorder=10, showfliers=False)
# plt.ylim([0.1, 1.1])
plt.tight_layout()
plt.savefig('beeswarm_type_GOterm' + current_time + '.png')
plt.savefig('beeswarm_type_GOterm' + current_time + '.pdf')
plt.close()



# #
