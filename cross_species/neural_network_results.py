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

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/standard_scaler_trained', 'rb') as f:
    scaler = pickle.load(f)


def open_input(species):
    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/%s.pkl' % species, 'rb') as f:
        yspecies, Xspecies, prot_rat, loc_species = pickle.load(f)
    Xspecies = Xspecies[:, 0:1024]
    Xspecies = scaler.transform(Xspecies)
    return Xspecies, yspecies, loc_species


Xmouse, ymouse, loc_mouse = open_input('mouse')
Xrat, yrat, loc_rat = open_input('rat')
Xhuman, yhuman, loc_human = open_input('human')
Xzebrafish, yzebrafish, loc_zebrafish = open_input('zebrafish')
Xcelegans, ycelegans, loc_celegans = open_input('celegans')
Xyeast, yyeast, loc_yeast = open_input('yeast')
Xathaliana, yathaliana, loc_athaliana = open_input('athaliana')


with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/human_index_setsX.pkl', 'rb') as f:
    Xhuman_train, Xhuman_valid, Xhuman_test = pickle.load(f)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/human_index_setsY.pkl', 'rb') as f:
    yhuman_train, yhuman_valid, yhuman_test = pickle.load(f)
Xhuman_test = Xhuman_test[:, 0:1024]
Xhuman_test = scaler.transform(Xhuman_test)


def get_indexes(species):
    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/protein_%s.pkl' % species, 'rb') as f:
        protein_indexes = pickle.load(f)
    protein_indexes.remove('GO:0003674')
    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/indexes/term_%s.pkl' % species, 'rb') as f:
        term_indexes = pickle.load(f)
    term_indexes.remove('GO:0003674')
    return protein_indexes, term_indexes


protein_indexes_mouse, term_indexes_mouse = get_indexes('mouse')
protein_indexes_rat, term_indexes_rat = get_indexes('rat')
protein_indexes_human_test, term_indexes_human_test = get_indexes('human_test')
protein_indexes_zebrafish, term_indexes_zebrafish = get_indexes('zebrafish')
protein_indexes_celegans, term_indexes_celegans = get_indexes('celegans')
protein_indexes_yeast, term_indexes_yeast = get_indexes('yeast')
protein_indexes_athaliana, term_indexes_athaliana = get_indexes('athaliana')


def fmax_threshold(Ytrue, Ypost1, t):
    Ytrue = Ytrue.transpose()
    Ypost1 = Ypost1.transpose()
    tokeep = np.where(np.sum(Ytrue, 0) > 0)[0]
    Ytrue = Ytrue[:, tokeep]
    Ypost1 = Ypost1[:, tokeep]

    _, rc, _, _ = precision_recall_fscore_support(Ytrue, (Ypost1 >= t).astype(int))
    rc_avg = np.mean(rc)

    tokeep = np.where(np.sum((Ypost1 >= t).astype(int), 0) > 0)[0]
    Ytrue_pr = Ytrue[:, tokeep]
    Ypost1_pr = Ypost1[:, tokeep]
    if tokeep.any():
        pr, _, _, _ = precision_recall_fscore_support(Ytrue_pr, (Ypost1_pr >= t).astype(int))
        pr_avg = np.mean(pr)
        coverage = len(pr) / len(rc)
    else:
        pr_avg = 0
        coverage = 0

    ff = (2 * pr_avg * rc_avg) / (pr_avg + rc_avg)

    return ff, coverage


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1024, 512, yhuman_train.shape[1]


# Create Tensors to hold inputs and outputs
def get_loader(Xdata, ydata):
    features = torch.from_numpy(Xdata)
    labels = torch.from_numpy(ydata)
    dataset_species = TensorDataset(features, labels)
    species_loader = DataLoader(dataset=dataset_species, batch_size=N, shuffle=True)
    return species_loader


loader_mouse = get_loader(Xmouse, ymouse)
loader_rat = get_loader(Xrat, yrat)
loader_human_test = get_loader(Xhuman_test, yhuman_test)
loader_zebrafish = get_loader(Xzebrafish, yzebrafish)
loader_celegans = get_loader(Xcelegans, ycelegans)
loader_yeast = get_loader(Xyeast, yyeast)
loader_athaliana = get_loader(Xathaliana, yathaliana)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class MLP(nn.Module):
    def __init__(self, input_dim=D_in, fc_dim=H, num_classes=D_out):
        super(MLP, self).__init__()

        # Define fully-connected layers and dropout
        self.layer1 = nn.Linear(input_dim, fc_dim)
        self.drop = nn.Dropout(p=0.3)
        self.layer2 = nn.Linear(fc_dim, num_classes)

    def forward(self, datax):
        x = datax

        # Compute fully-connected part and apply dropout
        x = F.relu(self.layer1(x))
        x = self.drop(x)
        outputforward = self.layer2(x)  # sigmoid in loss function

        return outputforward


model = MLP()
model1 = MLP()

########### for correction GO hierarchy
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GO_terms/GO_depth_all_mouse_proteins.pkl', 'rb') as fw:
    depth_terms = pickle.load(fw)

max_value = 0
terms_per_depth = {k: [] for k in range(20)}
for key in depth_terms.keys():
    depth = depth_terms[key][0]
    terms_per_depth[depth].append(key)
    if depth > max_value:
        max_value = depth

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/Seeds_random_state', 'rb') as f:
    seeds = pickle.load(f)

########### for protein centric
best_epoch = 34
threshold1 = 0.3

checkpoint = torch.load(
    '/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/epochs/epoch%s' % best_epoch)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()


def predictions_protein(loader_, indexes, loc, species):
    y_true = []
    y_pred_sigm = []
    with torch.no_grad():
        for data in loader_:
            X, y = data
            output = model.forward(X.float())
            y_true.append(y.cpu().numpy().squeeze())
            y_pred_sigm.append(torch.sigmoid(output).cpu().numpy().squeeze())

    # Calculate evaluation metrics
    y_true1 = np.vstack(y_true)
    y_pred_sigm1 = np.vstack(y_pred_sigm)

    # get the terms for fmax performance test
    y_true_protein = np.zeros((y_true1.shape[0], len(indexes)))
    y_pred_sigm_protein = np.zeros((y_true1.shape[0], len(indexes)))
    location = {}
    depth_lvl = np.zeros((len(indexes),))
    for i, key1 in enumerate(indexes):
        y_true_protein[:, i] = y_true1[:, loc[key1]]
        y_pred_sigm_protein[:, i] = y_pred_sigm1[:, loc_human[key1]]
        location[key1] = i
        depth_lvl[i] = depth_terms[key1][0]

    for depth1 in np.arange(max_value, -1, -1):
        for key11 in indexes:
            if key11 in terms_per_depth[depth1]:
                if depth_terms[key11][1]:
                    for up_terms in depth_terms[key11][1]:
                        if up_terms in indexes:
                            index = np.where(
                                y_pred_sigm_protein[:, location[key11]] < y_pred_sigm_protein[:, location[up_terms]])[0]
                            if index.any():
                                for protein in index:
                                    y_pred_sigm_protein[protein, location[key11]] = y_pred_sigm_protein[
                                        protein, location[up_terms]]

    # fmax score
    avg_fmax, cov = fmax_threshold(y_true_protein, y_pred_sigm_protein, threshold1)
    print('for %s' % species)
    print('f1-score: %s' % avg_fmax)
    print('coverage: %s' % cov)

    boot_results_f1 = np.zeros(len(seeds))

    for ij, seed in enumerate(seeds):
        Ypost, Yval = resample(y_pred_sigm_protein, y_true_protein, random_state=seed, stratify=y_true_protein)
        fmea = fmax_threshold(Yval, Ypost, threshold1)
        boot_results_f1[ij] = fmea[0]

    print('confidence interval')
    print(np.percentile(boot_results_f1, [2.5, 97.5]))
    sys.stdout.flush()

    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/predictions/predictions_protein_centric_%s.pkl' % species, 'wb') as fw:
        pickle.dump({'Yval': y_true_protein, 'Ypost': y_pred_sigm_protein, 'loc_term': location, 'allY': y_true1}, fw)



predictions_protein(loader_mouse, protein_indexes_mouse, loc_mouse, 'mouse')
predictions_protein(loader_rat, protein_indexes_rat, loc_rat, 'rat')
predictions_protein(loader_human_test, protein_indexes_human_test, loc_human, 'human_test')
predictions_protein(loader_zebrafish, protein_indexes_zebrafish, loc_zebrafish, 'zebrafish')
predictions_protein(loader_celegans, protein_indexes_celegans, loc_celegans, 'celegans')
predictions_protein(loader_yeast, protein_indexes_yeast, loc_yeast, 'yeast')
predictions_protein(loader_athaliana, protein_indexes_athaliana, loc_athaliana, 'athaliana')



########### for term centric
best_epoch = 32

checkpoint1 = torch.load('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/epochs/epoch%s' % best_epoch)
model1.load_state_dict(checkpoint1['model_state_dict'])

model1.eval()


def predictions_term(loader_, indexes, loc, species):
    y_true = []
    y_pred_sigm = []
    with torch.no_grad():
        for data in loader_:
            X, y = data
            output = model1.forward(X.float())
            y_true.append(y.cpu().numpy().squeeze())
            y_pred_sigm.append(torch.sigmoid(output).cpu().numpy().squeeze())

    # Calculate evaluation metrics
    y_true1 = np.vstack(y_true)
    y_pred_sigm1 = np.vstack(y_pred_sigm)

    # get the terms for rocauc performance test
    y_true_term = np.zeros((y_true1.shape[0], len(indexes)))
    y_pred_sigm_term = np.zeros((y_true1.shape[0], len(indexes)))
    location = {}
    depth_lvl = np.zeros((len(indexes),))
    for ii, key in enumerate(indexes):
        y_true_term[:, ii] = y_true1[:, loc[key]]
        y_pred_sigm_term[:, ii] = y_pred_sigm1[:, loc_human[key]]
        location[key] = ii
        depth_lvl[ii] = depth_terms[key][0]


    for depth1 in np.arange(max_value, -1, -1):
        for key11 in indexes:
            if key11 in terms_per_depth[depth1]:
                if depth_terms[key11][1]:
                    for up_terms in depth_terms[key11][1]:
                        if up_terms in indexes:
                            index = np.where(y_pred_sigm_term[:, location[key11]] < y_pred_sigm_term[:, location[up_terms]])[0]
                            if index.any():
                                for protein in index:
                                    y_pred_sigm_term[protein, location[key11]] = y_pred_sigm_term[protein, location[up_terms]]

    # ROC AUC score
    iii = np.where(np.sum(y_true_term, 0) > 0)[0]
    avg_rocauc = np.nanmean(roc_auc_score(y_true_term[:, iii], y_pred_sigm_term[:, iii], average=None))

    print('for %s' % species)
    print('rocauc: %s' % avg_rocauc)

    boot_results_rocauc = np.zeros(len(seeds))

    for ij, seed in enumerate(seeds):
        Ypost, Yval = resample(y_pred_sigm_term, y_true_term, random_state=seed, stratify=y_true_term)
        iiii = np.where(np.sum(Yval, 0) > 0)[0]
        boot_results_rocauc[ij] = np.nanmean(roc_auc_score(Yval[:, iiii], Ypost[:, iiii], average=None))

    print('confidence interval')
    print(np.percentile(boot_results_rocauc, [2.5, 97.5]))
    sys.stdout.flush()

    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/human_model/predictions/predictions_term_centric_%s.pkl' % species, 'wb') as fw:
        pickle.dump({'Yval': y_true_term, 'Ypost': y_pred_sigm_term, 'loc_term': location, 'allY': y_true1}, fw)



predictions_term(loader_mouse, term_indexes_mouse, loc_mouse, 'mouse')
predictions_term(loader_rat, term_indexes_rat, loc_rat, 'rat')
predictions_term(loader_human_test, term_indexes_human_test, loc_human, 'human_test')
predictions_term(loader_zebrafish, term_indexes_zebrafish, loc_zebrafish, 'zebrafish')
predictions_term(loader_celegans, term_indexes_celegans, loc_celegans, 'celegans')
predictions_term(loader_yeast, term_indexes_yeast, loc_yeast, 'yeast')
predictions_term(loader_athaliana, term_indexes_athaliana, loc_athaliana, 'athaliana')


#
# #### how wel do low training number proteins do cross organisms
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/mouse.pkl', 'rb') as f:
#     ymouse, Xm, prot_mouse, loc_mouse = pickle.load(f)
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/mouse_index_setsY.pkl', 'rb') as f:
#     ymouse_train, ymouse_valid, ymouse_test = pickle.load(f)
#
# index = np.where(np.sum(ymouse_train, axis=0) == 5)
# index1 = np.where(np.sum(ymouse_train, axis=0) == 6)
#
# test_terms = []
# for name, loc in loc_mouse.items():
#     if loc in index[0]:
#         test_terms.append(name)
#     if loc in index1[0]:
#         test_terms.append(name)
#
# def test(species):
#     with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/validation_results/predictions_term_centric_NN/%s.pkl' % species, 'rb') as fw:
#         data = pickle.load(fw)
#
#     rocauc_values = []
#     terms = []
#     for term in test_terms:
#         if term in data['loc_term']:
#             rocauc_values.append(roc_auc_score(data['Yval'][:, data['loc_term'][term]], data['Ypost'][:, data['loc_term'][term]]))
#             terms.append(term)
#
#     rocauc_values = np.array(rocauc_values)
#     avg = np.nanmean(rocauc_values)
#     return rocauc_values, terms, avg
#
# mouse_test_5 = test('mouse_test')
# rat_5 = test('rat')
# human_5 = test('human')
# zebrafish_5 = test('zebrafish')
# celegans_5 = test('celegans')
# yeast_5 = test('yeast')
# athaliana_5 = test('athaliana')
#
# elements_in_all = set(mouse_test_5[1]) & set(rat_5[1]) & set(human_5[1]) & set(zebrafish_5[1]) & set(celegans_5[1]) & set(yeast_5[1]) & set(athaliana_5[1])
# all_df_f = mouse_test_5[1] + rat_5[1] + human_5[1] + zebrafish_5[1] + celegans_5[1] +yeast_5[1] + athaliana_5[1]
# unique_f = list(set(func for func in all_df_f))
#
# elems_in_6 = []
# for ele1 in unique_f:
#     summie = np.zeros(7)
#     summie[0] = int(ele1 in mouse_test_5[1])
#     summie[1] = int(ele1 in rat_5[1])
#     summie[2] = int(ele1 in human_5[1])
#     summie[3] = int(ele1 in zebrafish_5[1])
#     summie[4] = int(ele1 in celegans_5[1])
#     summie[5] = int(ele1 in yeast_5[1])
#     summie[6] = int(ele1 in athaliana_5[1])
#     if np.sum(summie) >= 6:
#         elems_in_6.append(ele1)
#
# spies = []
# rocauc = []
# go_term = []
# def test_selected(species):
#     with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/validation_results/predictions_term_centric_NN/%s.pkl' % species, 'rb') as fw:
#         data = pickle.load(fw)
#
#     rocauc_values = []
#     toch_term = []
#     for term in elems_in_6:
#         if term in data['loc_term']:
#             rocauc_values.append(roc_auc_score(data['Yval'][:, data['loc_term'][term]], data['Ypost'][:, data['loc_term'][term]]))
#             toch_term.append(term)
#
#             spies.append(species)
#             rocauc.append(roc_auc_score(data['Yval'][:, data['loc_term'][term]], data['Ypost'][:, data['loc_term'][term]]))
#             go_term.append(term)
#
#     rocauc_values = np.array(rocauc_values)
#     avg = np.nanmean(rocauc_values)
#     return rocauc_values, avg
#
# mouse_test_55 = test_selected('mouse_test')
# rat_55 = test_selected('rat')
# human_55 = test_selected('human')
# zebrafish_55 = test_selected('zebrafish')
# celegans_55 = test_selected('celegans')
# yeast_55 = test_selected('yeast')
# athaliana_55 = test_selected('athaliana')
#
# df_in_all = pd.DataFrame({'Species': spies, 'GO term': go_term, 'Rocauc score': rocauc})
#
# now = datetime.now()
# current_time = now.strftime("%d%m%Y%H%M%S")
# plt.figure()
# sns.set(style="whitegrid")
# sns.color_palette("Paired")
# sns.swarmplot(y="Rocauc score", x="Species", hue='GO term', palette= 'Paired', data=df_in_all, size=6)
# plt.title('across species function performance')
# # sns.boxplot(y="GO_term", x="Rocauc score", data=df_in_all, order =ordah ,medianprops = {'color': 'gray', 'linewidth': 1}, capprops = {'color': 'gray', 'linewidth': 1}, whiskerprops = {'color': 'gray', 'linewidth': 1}, boxprops={'facecolor': 'None', 'edgecolor': 'gray', 'linewidth': 1}, showfliers=False)
# # plt.xlim([0.48, 1.02])
# plt.tight_layout()
# plt.savefig('beeswarm_5train' + current_time + '.png')
# plt.savefig('beeswarm_5train' + current_time + '.pdf')
# plt.close()