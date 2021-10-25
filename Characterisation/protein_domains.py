import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
import pandas as pd
import sys

domain_list = []
with open('./raw_structural_data/domains.tsv') as f:
    for line in f:
        domain_list.append(line.split("\t")[0])

family_list = []
with open('./raw_structural_data/families.tsv') as f:
    for line in f:
        family_list.append(line.split("\t")[0])

superfamily_list = []
with open('./raw_structural_data/superfamilies.tsv') as f:
    for line in f:
        superfamily_list.append(line.split("\t")[0])

domains = {}
familie = {}
superfamilie = {}
with open('./raw_structural_data/domains.tab') as f:
    for line in f:
        if len(line) > 15:
            id = line.split("\t")[0]
            string = line.split("\t")[1]

            index = [m.start() for m in re.finditer('IPR', string)]
            do = []
            fam = []
            supfam =[]
            for i in index:
                domain = ''
                for j in range(20):
                    if string[i + j] == ';':
                        if domain in domain_list:
                            do.append(domain)
                        if domain in family_list:
                            fam.append(domain)
                        if domain in superfamily_list:
                            supfam.append(domain)
                        break
                    if string[i+j] != '"':
                        domain = domain + string[i+j]
            if do:
                domains[id] = do
            if fam:
                familie[id] = fam
            if supfam:
                superfamilie[id] = supfam



testsetFile = sys.argv[1]
testProteinNamesFile = sys.argv[2]
modelPath = sys.argv[3]

with open(testsetFile, 'rb') as fw:
    Xtest, Ytest, GO_terms = pickle.load(fw)

with open(testProteinNamesFile, 'rb ') as f:
    protein_id_test = pickle.load(f)


with open(modelPath + '/performance.pkl', 'rb') as f:
    dd = pickle.load(f)
ave_term_1024 = dd['tc']


### for domains
num_domain = 0
ind_pro = []
for num, pro in enumerate(protein_id_test):
    if pro in domains.keys():
        num_domain = num_domain + 1
        ind_pro.append(num)
ind_pro = np.array(ind_pro)
print('number tested proteins: %s' % num_domain)


joe = 0
term = []
highest = []
len1 = []
fac_overlap = []
GO = {}
for ii in range(Ytest.shape[1]):
    test_pro = np.where(Ytest[:, ii] == 1)
    match = np.intersect1d(test_pro[0], ind_pro)


    if match.shape[0] / test_pro[0].shape[0] >= 0.5:
        GO[ii] = joe
        joe = joe + 1

        dom = []
        length = []
        for mm in match:
            dom.append(domains[protein_id_test[mm]])
            length.append(len(domains[protein_id_test[mm]]))
        flat_list = set([item for sublist in dom for item in sublist])
        length = np.array(length)

        prevalences = []
        for test_domain in flat_list:
            prev = 0
            for mm in match:
                if test_domain in domains[protein_id_test[mm]]:
                    prev = prev + 1
            prevalences.append(prev/match.shape[0])
        prevalences = np.array(prevalences)

        flat_list = list(flat_list)
        best_domain = flat_list[np.argmax(prevalences)]
        sum1 = 0
        for jj, protein in enumerate(protein_id_test):
            if protein in domains.keys():
                if not jj in match:
                    if best_domain in domains[protein]:
                        sum1 = sum1 + 1
        fac_protein = sum1/ (num_domain - len(match))

        term.append(ave_term_1024[ii])
        len1.append(np.mean(length))
        highest.append(np.max(prevalences))
        fac_overlap.append(fac_protein)
print('number tested terms: %s' % joe)

term = np.array(term)

from yellowbrick.regressor import CooksDistance
visualizer = CooksDistance()
visualizer.fit(term.reshape(-1,1), len1)
now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.savefig('figures/outliers' + current_time + '.png')

df = pd.DataFrame({'perf': term, 'avg': len1})
df.drop(df.loc[df['perf']==df.perf.max()].index, inplace=True) #dropping the row

perf_domain1 = stats.spearmanr(term, highest)[0]
perf_var = stats.spearmanr(df['perf'], df['avg'])[0]
perf_fac = stats.spearmanr(term, fac_overlap)[0]

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.plot(highest, term, 'o')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('shared domain')
plt.ylabel('rocauc')
plt.savefig('figures/domain50%' + current_time + '.png')
plt.savefig('figures/domain50%' + current_time + '.pdf')
plt.close()

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.plot(df['avg'], df['perf'], 'o')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('shared domain')
plt.ylabel('rocauc')
plt.savefig('figures/domain50%_numdomains' + current_time + '.png')
plt.savefig('figures/domain50%_numdomains' + current_time + '.pdf')
plt.close()

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.plot(fac_overlap, term, 'o')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('shared domain')
plt.ylabel('rocauc')
plt.savefig('figures/domain80%_overlap' + current_time + '.png')
plt.savefig('figures/domain80%_overlap' + current_time + '.pdf')
plt.close()





### for family
num_domain = 0
ind_pro = []
for num, pro in enumerate(protein_id_test):
    if pro in familie.keys():
        num_domain = num_domain + 1
        ind_pro.append(num)
ind_pro = np.array(ind_pro)
print('number tested proteins: %s' % num_domain)


joe = 0
term = []
highest1 = []
len1 = []
fac_overlap = []
GO1 = {}
for ii in range(Ytest.shape[1]):
    test_pro = np.where(Ytest[:, ii] == 1)
    match = np.intersect1d(test_pro[0], ind_pro)

    if match.shape[0] / test_pro[0].shape[0] >= 0.5:
        GO1[ii] = joe
        joe = joe + 1


        dom = []
        length = []
        for mm in match:
            dom.append(familie[protein_id_test[mm]])
            length.append(len(familie[protein_id_test[mm]]))
        flat_list = set([item for sublist in dom for item in sublist])
        length = np.array(length)

        prevalences = []
        for test_domain in flat_list:
            prev = 0
            for mm in match:
                if test_domain in familie[protein_id_test[mm]]:
                    prev = prev + 1
            prevalences.append(prev/match.shape[0])
        prevalences = np.array(prevalences)

        flat_list = list(flat_list)
        best_domain = flat_list[np.argmax(prevalences)]
        sum1 = 0
        for jj, protein in enumerate(protein_id_test):
            if protein in familie.keys():
                if not jj in match:
                    if best_domain in familie[protein]:
                        sum1 = sum1 + 1
        fac_protein = sum1/ (num_domain - len(match))

        term.append(ave_term_1024[ii])
        len1.append(np.mean(length))
        highest1.append(np.max(prevalences))
        fac_overlap.append(fac_protein)
print('number tested terms: %s' % joe)

term = np.array(term)



df = pd.DataFrame({'perf': term, 'avg': len1})
df.drop(df.loc[df['perf']==df.perf.max()].index, inplace=True) #dropping the row

perf_domain1 = stats.spearmanr(term, highest)[0]
perf_var = stats.spearmanr(df['perf'], df['avg'])[0]
perf_fac = stats.spearmanr(term, fac_overlap)[0]


perf_domain = stats.spearmanr(term, highest1)[0]
perf_var = stats.spearmanr(df['perf'], df['avg'])[0]
perf_fac = stats.spearmanr(term, fac_overlap)[0]

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.plot(highest1, term, 'o')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('shared domain')
plt.ylabel('rocauc')
plt.savefig('figures/family50%' + current_time + '.png')
plt.savefig('figures/family50%' + current_time + '.pdf')
plt.close()

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.plot(df['avg'], df['perf'], 'o')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('shared domain')
plt.ylabel('rocauc')
plt.savefig('figures/family50%_numfamily' + current_time + '.png')
plt.savefig('figures/family50%_numfamily' + current_time + '.pdf')
plt.close()

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.plot(fac_overlap, term, 'o')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('shared domain')
plt.ylabel('rocauc')
plt.savefig('figures/family80%_overlap' + current_time + '.png')
plt.savefig('figures/family80%_overlap' + current_time + '.pdf')
plt.close()
















### for superfamily
num_domain = 0
ind_pro = []
for num, pro in enumerate(protein_id_test):
    if pro in superfamilie.keys():
        num_domain = num_domain + 1
        ind_pro.append(num)
ind_pro = np.array(ind_pro)
print('number tested proteins: %s' % num_domain)


joe = 0
term = []
highest2 = []
len1 = []
fac_overlap = []
GO2 = {}
for ii in range(Ytest.shape[1]):
    test_pro = np.where(Ytest[:, ii] == 1)
    match = np.intersect1d(test_pro[0], ind_pro)

    if match.shape[0] / test_pro[0].shape[0] >= 0.5:
        GO2[ii] = joe
        joe = joe + 1

        dom = []
        length = []
        for mm in match:
            dom.append(superfamilie[protein_id_test[mm]])
            length.append(len(superfamilie[protein_id_test[mm]]))
        flat_list = set([item for sublist in dom for item in sublist])
        length = np.array(length)

        prevalences = []
        for test_domain in flat_list:
            prev = 0
            for mm in match:
                if test_domain in superfamilie[protein_id_test[mm]]:
                    prev = prev + 1
            prevalences.append(prev/match.shape[0])
        prevalences = np.array(prevalences)

        flat_list = list(flat_list)
        best_domain = flat_list[np.argmax(prevalences)]
        sum1 = 0
        for jj, protein in enumerate(protein_id_test):
            if protein in superfamilie.keys():
                if not jj in match:
                    if best_domain in superfamilie[protein]:
                        sum1 = sum1 + 1
        fac_protein = sum1/ (num_domain - len(match))

        term.append(ave_term_1024[ii])
        len1.append(np.mean(length))
        highest2.append(np.max(prevalences))
        fac_overlap.append(fac_protein)
print('number tested terms: %s' % joe)

perf_domain = stats.spearmanr(term, highest)[0]
perf_var = stats.spearmanr(term, len1)[0]
perf_fac = stats.spearmanr(term, fac_overlap)[0]

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.plot(highest, term, 'o')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('shared domain')
plt.ylabel('rocauc')
plt.savefig('figures/superfamily50%' + current_time + '.png')
plt.savefig('figures/superfamily50%' + current_time + '.pdf')
plt.close()

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.plot(len1, term, 'o')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('shared domain')
plt.ylabel('rocauc')
plt.savefig('figures/superfamily50%_numfamily' + current_time + '.png')
plt.savefig('figures/superfamily50%_numfamily' + current_time + '.pdf')
plt.close()

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.plot(fac_overlap, term, 'o')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('shared domain')
plt.ylabel('rocauc')
plt.savefig('figures/superfamily50%_overlap' + current_time + '.png')
plt.savefig('figures/superfamily50%_overlap' + current_time + '.pdf')
plt.close()


k = np.fromiter(GO.keys(), dtype=float)
k1 = np.fromiter(GO1.keys(), dtype=float)
k2 = np.fromiter(GO2.keys(), dtype=float)

hieh = np.intersect1d(k1, k)
new1=[]
new2=[]
for tt in hieh:
    new1.append(highest[GO[tt]])
    new2.append(highest1[GO1[tt]])

stats.spearmanr(new1, new2)

hieh = np.intersect1d(k2, k)
new1=[]
new2=[]
for tt in hieh:
    new1.append(highest[GO[tt]])
    new2.append(highest2[GO2[tt]])

stats.spearmanr(new1, new2)

hieh = np.intersect1d(k2, k1)
new1=[]
new2=[]
for tt in hieh:
    new1.append(highest1[GO1[tt]])
    new2.append(highest2[GO2[tt]])

stats.spearmanr(new1, new2)




with open('/somedirectory/GO_depth_all_mouse_proteins.pkl', 'rb') as fw:
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

# grouping terms
kids_terms = {}
for term in depth_2_terms:
    with open("/somedirectory/GO_descendants/%s.txt" % term) as f:
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
df_domain = []
df_family = []
df_superfam = []
df_overlap = []
for term1 in kids_terms.keys():
    avg = []
    avg.append(ave_term_1024[np.where(np.array(terms) == term1)[0][0]])
    if kids_terms[term1][1]:
        for kid in kids_terms[term1][1]:
            if kid in terms:
                avg.append(ave_term_1024[np.where(np.array(terms) == kid)[0][0]])
    if len(avg) >= 5:
        # avg_rocauc[term1] = mean(avg)
        # avg_rocauc_numkids[term1] = len(avg)

        for kid in kids_terms[term1][1]:
            if kid in terms:
                if kids_terms[term1][0] != 'heterocyclic compound binding':
                    df_rocauc.append(ave_term_1024[np.where(np.array(terms) == kid)[0][0]])
                    df_depth.append(int(depth_GO[kid][0]))
                    df_function.append(kids_terms[term1][0])
                    # df_change.append(improved[np.where(np.array(terms) == kid)[0][0]] - ave_term_1024[
                    #     np.where(np.array(terms) == kid)[0][0]])
                    if np.where(np.array(terms) == kid)[0][0] in GO.keys():
                        df_domain.append(highest[GO[np.where(np.array(terms) == kid)[0][0]]])
                    else:
                        df_domain.append(np.nan)
                    if np.where(np.array(terms) == kid)[0][0] in GO1.keys():
                        df_family.append(highest1[GO1[np.where(np.array(terms) == kid)[0][0]]])
                    else:
                        df_family.append(np.nan)
                    if np.where(np.array(terms) == kid)[0][0] in GO2.keys():
                        df_superfam.append(highest2[GO2[np.where(np.array(terms) == kid)[0][0]]])
                        df_overlap.append(fac_overlap[GO2[np.where(np.array(terms) == kid)[0][0]]])

                    else:
                        df_superfam.append(np.nan)
                        df_overlap.append(np.nan)

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

df = pd.DataFrame({'GO term': df_function, 'Rocauc': df_rocauc, 'Domain': df_domain, 'Family': df_family, 'Superfamily': df_superfam, 'Overlap': df_overlap})
kek = df.groupby('GO term').mean()
kek1 = df.groupby('GO term').size()
kek2 = df.groupby('GO term').median()
kek3 = df.groupby('GO term').var()

df_new = pd.DataFrame({'Roc': kek2['Rocauc'], 'Domain': kek['Domain'], 'Family': kek['Family'], 'Superfamily': kek['Superfamily'], 'Overlap': kek['Overlap']})

df_new.reindex(ordah)

def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(stats.spearmanr(df[r], df[c])[1], 4)
    return pvalues

calculate_pvalues(df_new)

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.plot(df_new['Overlap'], df_new['Roc'], 'o')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('overlap')
plt.ylabel('rocauc')
plt.savefig('figures/GOcategory_overlap' + current_time + '.pdf')
plt.close()
