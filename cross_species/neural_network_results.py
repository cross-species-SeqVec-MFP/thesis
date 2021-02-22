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

# this code evaluates the performance of the MLP on the target species

directory = '/somedirectory/fasta_GO_embeddings'

directory1 = '/somedirectory/evidence_codes'

root_term = 'GO:0005575' # change to root term for either BP, MF or CC

directory_epochs = '/somedirectory/neural_model/epochs'


with open('%s/standard_scaler_trained' % directory1, 'rb') as f:
    scaler = pickle.load(f)


def open_input(species):
    with open('%s/%s.pkl' % (directory, species), 'rb') as f:
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

with open('%s/mouse_index_setsX.pkl' % directory, 'rb') as f:
    Xmouse_train, Xmouse_valid, Xmouse_test = pickle.load(f)
with open('%s/mouse_index_setsY.pkl' % directory, 'rb') as f:
    ymouse_train, ymouse_valid, ymouse_test = pickle.load(f)
Xmouse_test = Xmouse_test[:, 0:1024]
Xmouse_test = scaler.transform(Xmouse_test)


def get_indexes(species):
    with open('%s/index_protein_centric/%s.pkl' % (directory1, species), 'rb') as f:
        protein_indexes = pickle.load(f)
    protein_indexes.remove(root_term)
    with open('%s/index_term_centric/%s.pkl' % (directory1, species), 'rb') as f:
        term_indexes = pickle.load(f)
    term_indexes.remove(root_term)
    return protein_indexes, term_indexes


protein_indexes_human, term_indexes_human = get_indexes('human')
protein_indexes_rat, term_indexes_rat = get_indexes('rat')
protein_indexes_mouse_test, term_indexes_mouse_test = get_indexes('mouse_test')
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
N, D_in, H, D_out = 64, 1024, 512, ymouse_train.shape[1]


# Create Tensors to hold inputs and outputs
def get_loader(Xdata, ydata):
    features = torch.from_numpy(Xdata)
    labels = torch.from_numpy(ydata)
    dataset_species = TensorDataset(features, labels)
    species_loader = DataLoader(dataset=dataset_species, batch_size=N, shuffle=True)
    return species_loader


loader_mouse_test = get_loader(Xmouse_test, ymouse_test)
loader_rat = get_loader(Xrat, yrat)
loader_human = get_loader(Xhuman, yhuman)
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
with open('%s/GO_levels_all_GOterms.pkl' % directory1, 'rb') as fw:
    depth_terms = pickle.load(fw)
del depth_terms[root_term]

max_value = 0
terms_per_depth = {k: [] for k in range(20)}
for key in depth_terms.keys():
    depth = depth_terms[key][0]
    terms_per_depth[depth].append(key)
    if depth > max_value:
        max_value = depth

with open('/somedirectory/Seeds_random_state', 'rb') as f:
    seeds = pickle.load(f)

with open('%s/valid_results' % directory_epochs, 'rb') as f:
    rocauc_scores, epochie, fmax_scores, coverage_scores, threshold_scores = pickle.load(f)


########### for protein centric
best_epoch = np.argmax(fmax_scores)
threshold1 = threshold_scores[np.argmax(fmax_scores)]

checkpoint = torch.load(
    '%s/epoch%s' % (directory_epochs,best_epoch))
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
        y_pred_sigm_protein[:, i] = y_pred_sigm1[:, loc_mouse[key1]]
        location[key1] = i
        depth_lvl[i] = depth_terms[key1][0]

    for depth1 in np.arange(max_value, -1, -1):
        for key11 in indexes:
            if key11 in terms_per_depth[depth1]:
                if depth_terms[key11][0]:
                    for up_terms in depth_terms[key11]:
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

    with open('%s/predictions_protein_centric_%s.pkl' % (directory_epochs, species), 'wb') as fw:
        pickle.dump({'Yval': y_true_protein, 'Ypost': y_pred_sigm_protein, 'loc_term': location, 'allY': y_true1}, fw)



predictions_protein(loader_mouse_test, protein_indexes_mouse_test, loc_mouse, 'mouse_test')
predictions_protein(loader_rat, protein_indexes_rat, loc_rat, 'rat')
predictions_protein(loader_human, protein_indexes_human, loc_human, 'human')
predictions_protein(loader_zebrafish, protein_indexes_zebrafish, loc_zebrafish, 'zebrafish')
predictions_protein(loader_celegans, protein_indexes_celegans, loc_celegans, 'celegans')
predictions_protein(loader_yeast, protein_indexes_yeast, loc_yeast, 'yeast')
predictions_protein(loader_athaliana, protein_indexes_athaliana, loc_athaliana, 'athaliana')



########### for term centric
best_epoch = np.argmax(rocauc_scores)

checkpoint1 = torch.load('%s/epoch%s' % (directory_epochs, best_epoch))
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
        y_pred_sigm_term[:, ii] = y_pred_sigm1[:, loc_mouse[key]]
        location[key] = ii
        depth_lvl[ii] = depth_terms[key][0]


    for depth1 in np.arange(max_value, -1, -1):
        for key11 in indexes:
            if key11 in terms_per_depth[depth1]:
                if depth_terms[key11][0]:
                    for up_terms in depth_terms[key11]:
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

    with open('%s/predictions_term_centric_%s.pkl' % (directory_epochs, species), 'wb') as fw:
        pickle.dump({'Yval': y_true_term, 'Ypost': y_pred_sigm_term, 'loc_term': location, 'allY': y_true1}, fw)



predictions_term(loader_mouse_test, term_indexes_mouse_test, loc_mouse, 'mouse_test')
predictions_term(loader_rat, term_indexes_rat, loc_rat, 'rat')
predictions_term(loader_human, term_indexes_human, loc_human, 'human')
predictions_term(loader_zebrafish, term_indexes_zebrafish, loc_zebrafish, 'zebrafish')
predictions_term(loader_celegans, term_indexes_celegans, loc_celegans, 'celegans')
predictions_term(loader_yeast, term_indexes_yeast, loc_yeast, 'yeast')
predictions_term(loader_athaliana, term_indexes_athaliana, loc_athaliana, 'athaliana')

