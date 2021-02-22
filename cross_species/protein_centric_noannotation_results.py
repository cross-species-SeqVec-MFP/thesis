import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F


def predictions(species):
    with open('/somedirectory/predictions_protein_centric_NN/%s_.pkl' % species, 'rb') as fw:
        data = pickle.load(fw)
    return data

mouse_test_pred = predictions('mouse_test')
rat_pred = predictions('rat')
human_pred = predictions('human')
zebrafish_pred = predictions('zebrafish')
yeast_pred = predictions('yeast')
athaliana_pred = predictions('athaliana')
celegans_pred = predictions('celegans')

def percentage_assigned(pred):
    binary = np.nonzero(pred['Ypost'] >= 0.3)
    binary1 = np.nonzero(pred['Ypost'] < 0.3)
    pred['Ypost'][binary] = 1
    pred['Ypost'][binary1] = 0

    perc = np.zeros((pred['Ypost'].shape[0]))
    for i, val in enumerate(perc):
        count = 0
        for j in range(pred['Ypost'].shape[1]):
            if pred['Ypost'][i, j] == 1:
                if pred['Yval'][i, j] == 1:
                    count = count + 1
        perc[i] = count / np.sum(pred['allY'][i, :])
    return perc

mouse_perc = percentage_assigned(mouse_test_pred)
print('mouse average annotations assigned: %s' % mouse_perc.mean())
print('mouse std: %s' % mouse_perc.std())
rat_perc = percentage_assigned(rat_pred)
print('rat average annotations assigned: %s' % rat_perc.mean())
print('rat std: %s' % rat_perc.std())
human_perc = percentage_assigned(human_pred)
print('human average annotations assigned: %s' % human_perc.mean())
print('human std: %s' % human_perc.std())
zebrafish_perc = percentage_assigned(zebrafish_pred)
print('zebrafish average annotations assigned: %s' % zebrafish_perc.mean())
print('zebrafish std: %s' % zebrafish_perc.std())
celegans_perc = percentage_assigned(celegans_pred)
print('celegans average annotations assigned: %s' % celegans_perc.mean())
print('celegans std: %s' % celegans_perc.std())
yeast_perc = percentage_assigned(yeast_pred)
print('yeast average annotations assigned: %s' % yeast_perc.mean())
print('yeast std: %s' % yeast_perc.std())
athaliana_perc = percentage_assigned(athaliana_pred)
print('athaliana average annotations assigned: %s' % athaliana_perc.mean())
print('athaliana std: %s' % athaliana_perc.std())


def hist_rocauc(perri, species):
    plt.hist(perri, weights=np.ones(len(perri)) / len(perri), bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], color='darkorange', alpha=1, rwidth=0.85)
    plt.axvline(perri.mean(), 0, 1, label='pyplot vertical line')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Percentage of total protein annotations assigned')
    plt.ylabel('Count')
    plt.ylim(0, 0.27)
    plt.savefig('Hist_percentage assigned%s' % species +'.png')
    plt.savefig('Hist_percentage assigned%s' % species +'.pdf')
    plt.close()

hist_rocauc(mouse_perc, 'mouse_test')
hist_rocauc(rat_perc, 'rat')
hist_rocauc(celegans_perc, 'celegans')
hist_rocauc(yeast_perc, 'yeast')
hist_rocauc(human_perc, 'human')
hist_rocauc(zebrafish_perc, 'zebrafish')
hist_rocauc(athaliana_perc, 'athaliana')



with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GO_terms/GO_depth_all_mouse_proteins.pkl', 'rb') as fw:
    depth_GO = pickle.load(fw)


def levels_assigned(pred):
    terms = {y:x for x,y in pred['loc_term'].items()}

    binary = np.nonzero(pred['Ypost'] >= 0.3)
    binary1 = np.nonzero(pred['Ypost'] < 0.3)
    pred['Ypost'][binary] = 1
    pred['Ypost'][binary1] = 0

    lvl = np.zeros((pred['Ypost'].shape[0]), dtype=object)
    for i, val in enumerate(lvl):
        list_lvl_trueonly = []
        for j in range(pred['Ypost'].shape[1]):
            if pred['Ypost'][i, j] == 1:
                if pred['Yval'][i, j] == 1:
                    if depth_GO[terms[j]][0] > 11:
                        list_lvl_trueonly.append(11)
                    else:
                        list_lvl_trueonly.append(depth_GO[terms[j]][0])
        lvl[i] = list_lvl_trueonly

    count_level_trueonly = np.zeros(11)
    for ii, level in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
        test_levels = np.linspace(level, 11, (12 - level))
        for pro in lvl:
            if np.any(np.in1d(test_levels, pro)):
                count_level_trueonly[ii] = count_level_trueonly[ii] + 1
    percentage_trueonly = count_level_trueonly / lvl.shape[0]
    return percentage_trueonly

mouse_levels = levels_assigned(mouse_test_pred)
rat_levels = levels_assigned(rat_pred)
human_levels = levels_assigned(human_pred)
zebrafish_levels = levels_assigned(zebrafish_pred)
celegans_levels = levels_assigned(celegans_pred)
yeast_levels = levels_assigned(yeast_pred)
athaliana_levels = levels_assigned(athaliana_pred)


print('true only predictions')
print(mouse_levels)
print(rat_levels)
print(human_levels)
print(zebrafish_levels)
print(celegans_levels)
print(yeast_levels)
print(athaliana_levels)

depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
depths = depth + depth + depth + depth + depth + depth + depth
species = ['mouse_test','mouse_test','mouse_test','mouse_test','mouse_test','mouse_test','mouse_test','mouse_test','mouse_test','mouse_test','mouse_test',
           'rat', 'rat', 'rat', 'rat', 'rat', 'rat', 'rat', 'rat', 'rat', 'rat', 'rat',
           'human', 'human', 'human', 'human', 'human', 'human', 'human', 'human', 'human', 'human', 'human',
           'zebrafish','zebrafish','zebrafish','zebrafish','zebrafish','zebrafish','zebrafish','zebrafish','zebrafish','zebrafish','zebrafish',
           'celegans','celegans','celegans','celegans','celegans','celegans','celegans','celegans','celegans','celegans','celegans',
           'yeast', 'yeast', 'yeast', 'yeast', 'yeast', 'yeast', 'yeast', 'yeast', 'yeast', 'yeast', 'yeast',
           'athaliana', 'athaliana','athaliana','athaliana','athaliana','athaliana','athaliana','athaliana','athaliana','athaliana','athaliana']
percentages = np.concatenate([mouse_levels, rat_levels, human_levels, zebrafish_levels, celegans_levels, yeast_levels, athaliana_levels])


df_in_all = pd.DataFrame({'Species': species, 'Depth': depths, 'Percentage': percentages})

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
plt.figure()
sns.set(style="whitegrid")
sns.swarmplot(y="Percentage", x="Depth", hue='Species', data=df_in_all, size=6)
plt.title('across species function performance')
plt.tight_layout()
plt.savefig('figures/percentageTP_species' + current_time + '.png')
plt.savefig('figures/percentageTP_species' + current_time + '.pdf')
plt.close()


#

# find the distributions of GO term depth
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GO_terms/GO_depth_all_mouse_proteins.pkl', 'rb') as fw:
    depth_GO = pickle.load(fw)

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/mouse_valid.pkl', 'rb') as f:
    mousevalid_indexes = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/mouse.pkl', 'rb') as f:
    mouse_index = pickle.load(f)

def GO_levels(index):
    levels = np.zeros((len(index),))
    for i, x in enumerate(index):
        levels[i] = int(depth_GO[x][0])
    return {'depth': levels}

mouse_train_pred = GO_levels(mouse_index)
mouse_valid_pred = GO_levels(mousevalid_indexes)

N = 11
ind = np.arange(N)    # the x locations for the groups
width = 0.35
labels = ['mouse train', 'mouse valid', 'mouse test', 'rat', 'human', 'zebrafish', 'c. elegans',
          'yeast', 'a. thaliana']

def prep_bar(pred):
    count_levels = np.zeros(11)
    for value in pred['depth']:
        if int(value) >= 11:
            value = 11
        count_levels[int(value)-1] = count_levels[int(value)-1] + 1
    count_levels = count_levels / len(pred['depth'])
    bottoms = np.zeros(11)
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        bottoms[i] = np.sum(count_levels[0:i])
    return count_levels, bottoms

all_levels = np.zeros((9, 11))
all_levels[0, :] = prep_bar(mouse_train_pred)[0]
all_levels[1, :] = prep_bar(mouse_valid_pred)[0]
all_levels[2, :] = prep_bar(mouse_test_pred)[0]
all_levels[3, :] = prep_bar(rat_pred)[0]
all_levels[4, :] = prep_bar(human_pred)[0]
all_levels[5, :] = prep_bar(zebrafish_pred)[0]
all_levels[6, :] = prep_bar(celegans_pred)[0]
all_levels[7, :] = prep_bar(yeast_pred)[0]
all_levels[8, :] = prep_bar(athaliana_pred)[0]
all_bottoms = np.zeros((9, 11))
all_bottoms[0, :] = prep_bar(mouse_train_pred)[1]
all_bottoms[1, :] = prep_bar(mouse_valid_pred)[1]
all_bottoms[2, :] = prep_bar(mouse_test_pred)[1]
all_bottoms[3, :] = prep_bar(rat_pred)[1]
all_bottoms[4, :] = prep_bar(human_pred)[1]
all_bottoms[5, :] = prep_bar(zebrafish_pred)[1]
all_bottoms[6, :] = prep_bar(celegans_pred)[1]
all_bottoms[7, :] = prep_bar(yeast_pred)[1]
all_bottoms[8, :] = prep_bar(athaliana_pred)[1]

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
fig, ax = plt.subplots()

ax.bar(labels, all_levels[:, 0], width, label='Depth 1')
for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    ax.bar(labels, all_levels[:, i], width, bottom=all_bottoms[:, i], label='Depth %s' % str(i+1))

plt.gcf().subplots_adjust(bottom=0.20)
plt.gcf().subplots_adjust(right=0.8)
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.75)
plt.ylabel('Fraction of total tested terms')
plt.title('Distribution of depth GO terms')
plt.savefig('stacked_barplot_depthGO' + current_time + '.png')
plt.savefig('stacked_barplot_depthGO' + current_time + '.pdf')
plt.close()
