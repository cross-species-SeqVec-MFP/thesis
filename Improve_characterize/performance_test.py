import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from statistics import mean
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit

Cs = [1]
reg = 'l2'
index = np.arange(0, 89, 1)
features = [1024, 3072, 5120]  # , 10240]
boot = np.arange(0, 100, 1)
perf1024 = np.zeros((441, len(boot)))
perf3072 = np.zeros((441, len(boot)))
perf5120 = np.zeros((441, len(boot)))
perf10240 = np.zeros((441, len(boot)))
terms = np.zeros((441,)).tolist()

terms2keep = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/lists/termIndicesToUse.npy')

C0_fea1024, C0_fea3072, C0_fea5120, C0_fea10240 = {}, {}, {}, {}
dict = []
dict.append([C0_fea1024, C0_fea3072, C0_fea5120, C0_fea10240])
dict = np.array(dict)

for g, C in enumerate(Cs):
    for h, fea in enumerate(features):
        for i in index:
            Nterms = 5
            lowerindex = i * Nterms
            upperindex = (i + 1) * Nterms
            if upperindex > len(terms2keep):
                upperindex = len(terms2keep)

            for o in boot:
                with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/'
                          'predictions/gen_mean/test/c%s_terms%s_%s_reg_%s_fea_%s_boot%s.pkl' %
                          (C, lowerindex, upperindex, reg, fea, o), 'rb') as fw:
                    data = pickle.load(fw)

                if fea == 1024:
                    if data['Ytest'].ndim == 2:
                        samples = data['Ytest'].shape[1]
                        a = np.arange(0, samples, 1)
                        for j in a:
                            ytrue = data['Ytest'][:, j]
                            ypred = data['Yprob'][j][:, 1]
                            perf1024[j + lowerindex, o] = roc_auc_score(ytrue, ypred)
                            terms[j + lowerindex] = data['GO terms'][j]
                    else:
                        ytrue = data['Ytest'][:]
                        ypred = data['Yprob'][:, 1]
                        perf1024[lowerindex, o] = roc_auc_score(ytrue, ypred)
                        terms[lowerindex] = data['GO terms']

                if fea == 3072:
                    if data['Ytest'].ndim == 2:
                        samples = data['Ytest'].shape[1]
                        a = np.arange(0, samples, 1)
                        for j in a:
                            ytrue = data['Ytest'][:, j]
                            ypred = data['Yprob'][j][:, 1]
                            perf3072[j + lowerindex, o] = roc_auc_score(ytrue, ypred)
                            terms[j + lowerindex] = data['GO terms'][j]
                    else:
                        ytrue = data['Ytest'][:]
                        ypred = data['Yprob'][:, 1]
                        perf3072[lowerindex, o] = roc_auc_score(ytrue, ypred)
                        terms[lowerindex] = data['GO terms']

                if fea == 5120:
                    if data['Ytest'].ndim == 2:
                        samples = data['Ytest'].shape[1]
                        a = np.arange(0, samples, 1)
                        for j in a:
                            ytrue = data['Ytest'][:, j]
                            ypred = data['Yprob'][j][:, 1]
                            perf5120[j + lowerindex, o] = roc_auc_score(ytrue, ypred)
                            terms[j + lowerindex] = data['GO terms'][j]
                    else:
                        ytrue = data['Ytest'][:]
                        ypred = data['Yprob'][:, 1]
                        perf5120[lowerindex, o] = roc_auc_score(ytrue, ypred)
                        terms[lowerindex] = data['GO terms']

                if fea == 10240:
                    if data['Ytest'].ndim == 2:
                        samples = data['Ytest'].shape[1]
                        a = np.arange(0, samples, 1)
                        for j in a:
                            ytrue = data['Ytest'][:, j]
                            ypred = data['Yprob'][j][:, 1]
                            perf10240[j + lowerindex, o] = roc_auc_score(ytrue, ypred)
                            terms[j + lowerindex] = data['GO terms'][j]
                    else:
                        ytrue = data['Ytest'][:]
                        ypred = data['Yprob'][:, 1]
                        perf10240[lowerindex, o] = roc_auc_score(ytrue, ypred)
                        terms[lowerindex] = data['GO terms']

with open(
        '/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/predictions/gen_mean/test/results.pkl',
        'wb') as fw:
    pickle.dump([perf1024, perf3072, perf5120, perf10240, terms], fw)

with open(
        '/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/predictions/gen_mean/test/results.pkl',
        'rb') as fw:
    perf1024, perf3072, perf5120, perf10240, terms = pickle.load(fw)

print('for 1024 fea and c is %s' % Cs[0])
overall_mcc = np.mean(perf1024, axis=0)
print('performance rocauc = %.4f, lower bound %.4f, upper bound %.4f' % (
np.mean(overall_mcc), np.percentile(overall_mcc, [2.5, 97.5][0]), np.percentile(overall_mcc, [2.5, 97.5][1])))

print('for 3072 fea and c is %s' % Cs[0])
overall_mcc = np.mean(perf3072, axis=0)
print('performance rocauc = %.4f, lower bound %.4f, upper bound %.4f' % (
np.mean(overall_mcc), np.percentile(overall_mcc, [2.5, 97.5][0]), np.percentile(overall_mcc, [2.5, 97.5][1])))

print('for 5120 fea and c is %s' % Cs[0])
overall_mcc = np.mean(perf5120, axis=0)
print('performance rocauc = %.4f, lower bound %.4f, upper bound %.4f' % (
np.mean(overall_mcc), np.percentile(overall_mcc, [2.5, 97.5][0]), np.percentile(overall_mcc, [2.5, 97.5][1])))

# print('for 10240 fea and c is %s' % Cs[0])
# overall_mcc = np.mean(perf10240, axis=0)
# print('performance rocauc = %.4f, lower bound %.4f, upper bound %.4f' % (np.mean(overall_mcc), np.percentile(overall_mcc, [2.5, 97.5][0]), np.percentile(overall_mcc, [2.5, 97.5][1])))
#
#



#############################################
# statistical testing
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/predictions/gen_mean/test/results.pkl', 'rb') as fw:
    perf1024, perf3072, perf5120, perf10240, terms = pickle.load(fw)

ave_term_1024 = np.mean(perf1024, axis=1)

####### best performing genmean 10240 features
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/generalized_means/Trained_LR/c%s_LR_numfea%s_1024LSTM1_test.pkl' % (1, 10240), 'rb') as fw:
    data1 = pickle.load(fw)
improved = roc_auc_score(data1['Yval'], data1['Ypost'], average=None)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GO_terms/GO_depth_all_mouse_proteins.pkl', 'rb') as fw:
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

df = pd.DataFrame({'GO term': df_function, 'Change rocauc score': df_change, 'GO term depth': df_depth})

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.figure(figsize=[9, 10])
sns.set(style="whitegrid")
sns.swarmplot(y="GO term", x="Change rocauc score", hue='GO term depth', zorder=1, data=df, size=3, order=ordah)
sns.boxplot(y="GO term", x="Change rocauc score", data=df, order=ordah, boxprops={'facecolor': 'None', "zorder": 10},
            zorder=10, showfliers=False)
# plt.ylim([0.1, 1.1])
plt.tight_layout()
plt.savefig('beeswarm_type_GOterm' + current_time + '.pdf')
plt.close()

# #


with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/Scaled_data/XYdata_moment_train_Yonly','rb') as fw:
    Ytrain = pickle.load(fw)

terms2keep = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/lists/termIndicesToUse.npy')
nb_TP = np.sum(Ytrain, axis=0)
sum_Y = nb_TP[terms2keep]
hist = np.digitize(sum_Y, [40, 200, 360, 520, 680, 840, 1000, 1160, 1320, 37000])

df = pd.DataFrame({'Amount of training samples': hist.astype(int), 'Rocauc score': ave_term_1024})
 
from scipy import stats
stats.spearmanr(sum_Y, ave_term_1024)
stats.spearmanr(levels, ave_term_1024)
stats.spearmanr(levels, sum_Y)

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
######### performance per GO term, grouped per amount of training samples
sns.set(style="whitegrid")
sns.swarmplot(x="Amount of training samples", y="Rocauc score", zorder=1, data=df,
              size=3)
sns.boxplot(x="Amount of training samples", y="Rocauc score", data=df, boxprops={'facecolor': 'None', "zorder": 10},
            zorder=10, showfliers=False)
plt.ylim([0.1, 1.1])
plt.savefig('beeswarm_num_training_bigdata' + current_time + '.pdf')
plt.close()

df = pd.DataFrame(
    {'Amount of training samples': hist.astype(int), 'Change rocauc score': improved - ave_term_1024})

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
######### change in rocauc performance per GO term, grouped per number of training samples
sns.set(style="whitegrid")
sns.swarmplot(x="Amount of training samples", y="Change rocauc score", zorder=1, data=df,
              size=3)  # , order=[1, 2, 3, 4, 5, 6, 7, 8, 9])
sns.boxplot(x="Amount of training samples", y="Change rocauc score", data=df,
            boxprops={'facecolor': 'None', "zorder": 10}, zorder=10, showfliers=False)
plt.ylim([-0.18, 0.25])
plt.savefig('beeswarm_num_training_change_rocauc_bigdata' + current_time + '.pdf')
plt.close()


df = pd.DataFrame({'GO annotation depth': levels.astype(int), 'Rocauc score': ave_term_1024})

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
######### performance per GO term, grouped per GO depth
sns.set(style="whitegrid")
sns.swarmplot(x="GO annotation depth", y="Rocauc score", zorder=1, data=df, size=3, order=[1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.title('LR classifier term-centric performance')
sns.boxplot(x="GO annotation depth", y="Rocauc score", data=df, boxprops={'facecolor': 'None', "zorder": 10}, zorder=10,
            showfliers=False)
plt.ylim([0.1, 1.1])
plt.savefig('beeswarm_GO_level_bigdata' + current_time + '.png')
plt.savefig('beeswarm_GO_level_bigdata' + current_time + '.pdf')
plt.close()

df = pd.DataFrame({'GO annotation depth': levels.astype(int), 'Change rocauc score': improved - ave_term_1024})

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
######### change in rocauc performance per GO term, grouped per GO depth
sns.set(style="whitegrid")
sns.swarmplot(x="GO annotation depth", y="Change rocauc score", zorder=1, data=df, size=3,
              order=[1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.title('LR classifier term-centric performance')
sns.boxplot(x="GO annotation depth", y="Change rocauc score", data=df, boxprops={'facecolor': 'None', "zorder": 10},
            zorder=10, showfliers=False)
plt.ylim([-0.18, 0.25])
plt.savefig('beeswarm_GO_level_change_rocauc_bigdata' + current_time + '.pdf')
plt.close()

##########################
# perf vs number of training samples
with open(
        '/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/Scaled_data/XYdata_moment_train_Yonly',
        'rb') as fw:
    Ytrain = pickle.load(fw)

terms2keep = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/lists/termIndicesToUse.npy')
nb_TP = np.sum(Ytrain, axis=0)
sum_Y = nb_TP[terms2keep]
np.histogram(sum_Y, [40, 100, 200, 300, 400, 500, 750, 1000, 2500, 37000])
now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")

plt.hlines(np.mean(ave_term_5120 - ave_term_1024), 0, 36000, zorder=1)
plt.scatter(sum_Y, ave_term_5120 - ave_term_1024, zorder=2, s=2, color='darkorange', alpha=0.75)
plt.grid(alpha=0.75)
plt.xlabel('number of TP in train')
plt.ylabel('Change rocauc performance')
plt.xlim((0, 36000))
# plt.ylim((-0.18, 0.18))
plt.title('performance per amount of training samples')
plt.savefig('figures/scatter_change_rocauc_TP' + current_time + '.png')
plt.savefig('figures/scatter_change_rocauc_TP' + current_time + '.pdf')
plt.close()


def scatter_change(baseline, change, num_fea, fea_type):
    now = datetime.now()
    current_time = now.strftime("%d%m%Y%H%M%S")

    plt.hlines(np.mean(change), 0.4, 1)
    plt.scatter(baseline, change, s=2, color='darkorange', alpha=0.75)
    plt.grid(alpha=0.75)
    plt.xlabel('Rocauc score baseline')
    plt.ylabel('Change in rocauc performance')
    plt.xlim((0.4, 1))
    plt.ylim((-0.18, 0.18))
    plt.title('Change in performance Logistic Regression classifier\n'
              'going from 1024 to %s %s features' % (num_fea, fea_type))
    plt.savefig('figures/scatter_change_rocauc_ELMo_%s%s' % (num_fea, fea_type) + current_time + '.png')
    plt.savefig('figures/scatter_change_rocauc_ELMo_%s%s' % (num_fea, fea_type) + current_time + '.pdf')
    plt.close()


scatter_change(ave_term_1024, ave_term_5120 - ave_term_1024, '5120', 'generalized mean')
scatter_change(ave_term_1024, ave_term_3072 - ave_term_1024, '3072', 'generalized mean')
scatter_change(ave_term_1024, ave_term_5120m - ave_term_1024m, '5120', 'moments')
scatter_change(ave_term_1024, ave_term_3072m - ave_term_1024m, '3072', 'moments')
scatter_change(ave_term_1024, ave_term_10240m - ave_term_1024m, '10240', 'moments')

import statsmodels.stats.weightstats as sm

sm.ztest(ave_term_5120 - ave_term_1024, x2=None, value=0, alternative='two-sided', usevar='pooled', ddof=1.0)
sm.ztest(ave_term_3072 - ave_term_1024, x2=None, value=0, alternative='two-sided', usevar='pooled', ddof=1.0)

sm.ztest(ave_term_3072m - ave_term_1024m, x2=None, value=0, alternative='two-sided', usevar='pooled', ddof=1.0)
sm.ztest(ave_term_5120m - ave_term_1024m, x2=None, value=0, alternative='two-sided', usevar='pooled', ddof=1.0)
sm.ztest(ave_term_10240m - ave_term_1024m, x2=None, value=0, alternative='two-sided', usevar='pooled', ddof=1.0)

from scipy import stats
stats.ttest_1samp(ave_term_5120 - ave_term_1024, 0)
stats.ttest_1samp(ave_term_3072 - ave_term_1024, 0)

stats.ttest_1samp(ave_term_3072m - ave_term_1024m, 0)
stats.ttest_1samp(ave_term_5120m - ave_term_1024m, 0)
stats.ttest_1samp(ave_term_10240m - ave_term_1024m, 0)

# import scipy.stats as ss
# ss.ttest_1samp(ave_term_5120 - ave_term_1024, 0)

def plot_change(change, num_fea, fea_type):
    now = datetime.now()
    current_time = now.strftime("%d%m%Y%H%M%S")

    plt.hist(change, bins=np.linspace(-0.18, 0.18, 45), color='darkorange', alpha=1, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Change in rocauc performance')
    plt.ylabel('Count')
    plt.xlim((-0.18, 0.18))
    plt.title('Change in performance Logistic Regression classifier\n'
              'going from 1024 to %s %s features' % (num_fea, fea_type))
    plt.savefig('figures/Hist_change_rocauc_ELMo_%s%s' % (num_fea, fea_type) + current_time + '.png')
    plt.savefig('figures/Hist_change_rocauc_ELMo_%s%s' % (num_fea, fea_type) + current_time + '.pdf')
    plt.close()
 

plot_change(ave_term_5120 - ave_term_1024, '5120', 'generalized mean')
plot_change(ave_term_3072 - ave_term_1024, '3072', 'generalized mean')
plot_change(ave_term_5120m - ave_term_1024m, '5120', 'moments')
plot_change(ave_term_10240m - ave_term_1024m, '10240', 'moments')
plot_change(ave_term_3072m - ave_term_1024m, '3072', 'moments')



df = pd.DataFrame({'GO annotation depth': levels.astype(int), 'Rocauc score': ave_term_1024})

def func_log(t, a, b):
    return a+b*np.log(t)

def func_lin(t, a, b):
    return a+b*t

# popt, pcov = curve_fit(func_lin,  df['GO annotation depth'],  df["Rocauc score"])
# dfafter = pd.DataFrame({'GO annotation depth': levels.astype(int), 'Fit': func_lin(df["GO annotation depth"], popt[0], popt[1])})
popt, pcov = curve_fit(func_lin,  [1, 2, 3, 4, 5, 6, 7, 8, 9],  df.groupby("Rocauc score").mean())
dfafter = pd.DataFrame({'GO annotation depth': levels.astype(int), 'Fit': func_lin(df["GO annotation depth"], popt[0], popt[1])})


now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.plot(df["GO annotation depth"], df["Rocauc score"], 'o', 'b-', label='data')
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], dfafter.groupby('GO annotation depth').mean(), 'r-', label='fit')
plt.legend()

plt.ylim([0.1, 1.1])
plt.savefig('fit_GO_level_bigdata' + current_time + '.png')
plt.savefig('fit_GO_level_bigdata' + current_time + '.pdf')
plt.close()

ss_res = np.sum((df["Rocauc score"] - dfafter["Fit"]) ** 2)
ss_tot = np.sum((df["Rocauc score"] - np.mean(df["Rocauc score"])) ** 2)

# r-squared
r2 = 1 - (ss_res / ss_tot)



