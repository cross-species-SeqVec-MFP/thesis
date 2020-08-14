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

# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
#
# 
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/'
#           'standard_scaler_trained_proteincentric.pkl', 'rb') as f:
#     scaler = pickle.load(f)
#
#
# def open_input(species):
#     with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/%s.pkl' % species, 'rb') as f:
#         yspecies, Xspecies, prot_rat, loc_species = pickle.load(f)
#     Xspecies = Xspecies[:, 0:1024]
#     Xspecies = scaler.transform(Xspecies)
#     return Xspecies, yspecies, loc_species
#
#
# Xrat, yrat, loc_rat = open_input('rat')
# Xhuman, yhuman, loc_human = open_input('human')
# Xzebrafish, yzebrafish, loc_zebrafish = open_input('zebrafish')
# Xcelegans, ycelegans, loc_celegans = open_input('celegans')
# Xyeast, yyeast, loc_yeast = open_input('yeast')
# Xathaliana, yathaliana, loc_athaliana = open_input('athaliana')
#
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/mouse.pkl', 'rb') as f:
#     ymouse, Xm, prot_mouse, loc_mouse = pickle.load(f)
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/mouse_index_setsX.pkl', 'rb') as f:
#     Xmouse_train, Xmouse_valid, Xmouse_test = pickle.load(f)
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/ylabels/mouse_index_setsY.pkl', 'rb') as f:
#     ymouse_train, ymouse_valid, ymouse_test = pickle.load(f)
# Xmouse_test = Xmouse_test[:, 0:1024]
# Xmouse_test = scaler.transform(Xmouse_test)
#
#
# def get_indexes(species):
#     with open(
#             '/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/%s.pkl' % species,
#             'rb') as f:
#         protein_indexes = pickle.load(f)
#     protein_indexes.remove('GO:0003674')
#     with open(
#             '/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_term_centric/%s.pkl' % species,
#             'rb') as f:
#         term_indexes = pickle.load(f)
#     term_indexes.remove('GO:0003674')
#     return protein_indexes, term_indexes
#
#
# protein_indexes_mouse_test, term_indexes_mouse_test = get_indexes('mouse_test')
# protein_indexes_rat, term_indexes_rat = get_indexes('rat')
# protein_indexes_human, term_indexes_human = get_indexes('human')
# protein_indexes_zebrafish, term_indexes_zebrafish = get_indexes('zebrafish')
# protein_indexes_celegans, term_indexes_celegans = get_indexes('celegans')
# protein_indexes_yeast, term_indexes_yeast = get_indexes('yeast')
# protein_indexes_athaliana, term_indexes_athaliana = get_indexes('athaliana')
#
#
# def fmax_threshold(Ytrue, Ypost1, t):
#     Ytrue = Ytrue.transpose()
#     Ypost1 = Ypost1.transpose()
#     tokeep = np.where(np.sum(Ytrue, 0) > 0)[0]
#     Ytrue = Ytrue[:, tokeep]
#     Ypost1 = Ypost1[:, tokeep]
#
#     _, rc, _, _ = precision_recall_fscore_support(Ytrue, (Ypost1 >= t).astype(int))
#     rc_avg = np.mean(rc)
#
#     tokeep = np.where(np.sum((Ypost1 >= t).astype(int), 0) > 0)[0]
#     Ytrue_pr = Ytrue[:, tokeep]
#     Ypost1_pr = Ypost1[:, tokeep]
#     if tokeep.any():
#         pr, _, _, _ = precision_recall_fscore_support(Ytrue_pr, (Ypost1_pr >= t).astype(int))
#         pr_avg = np.mean(pr)
#         coverage = len(pr) / len(rc)
#     else:
#         pr_avg = 0
#         coverage = 0
#
#     ff = (2 * pr_avg * rc_avg) / (pr_avg + rc_avg)
#
#     return ff, coverage
#
#
# # N is batch size; D_in is input dimension;
# # H is hidden dimension; D_out is output dimension.
# N, D_in, H, D_out = 64, 1024, 512, ymouse_train.shape[1]
#
#
# # Create Tensors to hold inputs and outputs
# def get_loader(Xdata, ydata):
#     features = torch.from_numpy(Xdata)
#     labels = torch.from_numpy(ydata)
#     dataset_species = TensorDataset(features, labels)
#     species_loader = DataLoader(dataset=dataset_species, batch_size=N, shuffle=True)
#     return species_loader
#
#
# loader_mouse_test = get_loader(Xmouse_test, ymouse_test)
# loader_rat = get_loader(Xrat, yrat)
# loader_human = get_loader(Xhuman, yhuman)
# loader_zebrafish = get_loader(Xzebrafish, yzebrafish)
# loader_celegans = get_loader(Xcelegans, ycelegans)
# loader_yeast = get_loader(Xyeast, yyeast)
# loader_athaliana = get_loader(Xathaliana, yathaliana)
#
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#
#
# class MLP(nn.Module):
#     def __init__(self, input_dim=D_in, fc_dim=H, num_classes=D_out):
#         super(MLP, self).__init__()
#
#         # Define fully-connected layers and dropout
#         self.layer1 = nn.Linear(input_dim, fc_dim)
#         self.drop = nn.Dropout(p=0.3)
#         self.layer2 = nn.Linear(fc_dim, num_classes)
#
#     def forward(self, datax):
#         x = datax
#
#         # Compute fully-connected part and apply dropout
#         x = F.relu(self.layer1(x))
#         x = self.drop(x)
#         outputforward = self.layer2(x)  # sigmoid in loss function
#
#         return outputforward
#
#
# model = MLP()
# model1 = MLP()
#
# ########### for correction GO hierarchy
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GO_terms/GO_depth_all_mouse_proteins.pkl', 'rb') as fw:
#     depth_terms = pickle.load(fw)
#
# max_value = 0
# terms_per_depth = {k: [] for k in range(20)}
# for key in depth_terms.keys():
#     depth = depth_terms[key][0]
#     terms_per_depth[depth].append(key)
#     if depth > max_value:
#         max_value = depth
#
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Proof_moments/optimize/Seeds_random_state', 'rb') as f:
#     seeds = pickle.load(f)
#
# ########### for protein centric
# best_epoch = 40
# threshold1 = 0.3
#
# checkpoint = torch.load(
#     '/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/NN_trained_epoch/epoch%s' % best_epoch)
# model.load_state_dict(checkpoint['model_state_dict'])
#
# model.eval()
#
#
# def predictions_protein(loader_, indexes, loc, species):
#     y_true = []
#     y_pred_sigm = []
#     with torch.no_grad():
#         for data in loader_:
#             X, y = data
#             output = model.forward(X.float())
#             y_true.append(y.cpu().numpy().squeeze())
#             y_pred_sigm.append(torch.sigmoid(output).cpu().numpy().squeeze())
#
#     # Calculate evaluation metrics
#     y_true1 = np.vstack(y_true)
#     y_pred_sigm1 = np.vstack(y_pred_sigm)
#
#     # get the terms for fmax performance test
#     y_true_protein = np.zeros((y_true1.shape[0], len(indexes)))
#     y_pred_sigm_protein = np.zeros((y_true1.shape[0], len(indexes)))
#     location = {}
#     depth_lvl = np.zeros((len(indexes),))
#     for i, key1 in enumerate(indexes):
#         y_true_protein[:, i] = y_true1[:, loc[key1]]
#         y_pred_sigm_protein[:, i] = y_pred_sigm1[:, loc_mouse[key1]]
#         location[key1] = i
#         depth_lvl[i] = depth_terms[key1][0]
#
#     for depth1 in np.arange(max_value, -1, -1):
#         for key11 in indexes:
#             if key11 in terms_per_depth[depth1]:
#                 if depth_terms[key11][1]:
#                     for up_terms in depth_terms[key11][1]:
#                         if up_terms in indexes:
#                             index = np.where(
#                                 y_pred_sigm_protein[:, location[key11]] < y_pred_sigm_protein[:, location[up_terms]])[0]
#                             if index.any():
#                                 for protein in index:
#                                     y_pred_sigm_protein[protein, location[key11]] = y_pred_sigm_protein[
#                                         protein, location[up_terms]]
#
#     with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/validation_results/predictions_protein_centric_NN/%s_try2.pkl' % species, 'wb') as fw:
#         pickle.dump({'Yval': y_true_protein, 'Ypost': y_pred_sigm_protein, 'allY': y_true1,'loc_term': location}, fw)
#
#
# predictions_protein(loader_mouse_test, protein_indexes_mouse_test, loc_mouse, 'mouse_test')
# predictions_protein(loader_rat, protein_indexes_rat, loc_rat, 'rat')
# predictions_protein(loader_human, protein_indexes_human, loc_human, 'human')
# predictions_protein(loader_zebrafish, protein_indexes_zebrafish, loc_zebrafish, 'zebrafish')
# predictions_protein(loader_celegans, protein_indexes_celegans, loc_celegans, 'celegans')
# predictions_protein(loader_yeast, protein_indexes_yeast, loc_yeast, 'yeast')
# predictions_protein(loader_athaliana, protein_indexes_athaliana, loc_athaliana, 'athaliana')
#




def predictions(species):
    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/validation_results/predictions_protein_centric_NN/%s_try2.pkl' % species, 'rb') as fw:
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

#
#
# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GO_terms/GO_depth_all_mouse_proteins.pkl', 'rb') as fw:
#     depth_GO = pickle.load(fw)
#
#
# def levels_assigned(pred):
#     terms = {y:x for x,y in pred['loc_term'].items()}
#
#     binary = np.nonzero(pred['Ypost'] >= 0.3)
#     binary1 = np.nonzero(pred['Ypost'] < 0.3)
#     pred['Ypost'][binary] = 1
#     pred['Ypost'][binary1] = 0
#
#     lvl = np.zeros((pred['Ypost'].shape[0]), dtype=object)
#     for i, val in enumerate(lvl):
#         list_lvl_trueonly = []
#         for j in range(pred['Ypost'].shape[1]):
#             if pred['Ypost'][i, j] == 1:
#                 if pred['Yval'][i, j] == 1:
#                     if depth_GO[terms[j]][0] > 11:
#                         list_lvl_trueonly.append(11)
#                     else:
#                         list_lvl_trueonly.append(depth_GO[terms[j]][0])
#         lvl[i] = list_lvl_trueonly
#
#     count_level_trueonly = np.zeros(11)
#     for ii, level in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
#         test_levels = np.linspace(level, 11, (12 - level))
#         for pro in lvl:
#             if np.any(np.in1d(test_levels, pro)):
#                 count_level_trueonly[ii] = count_level_trueonly[ii] + 1
#     percentage_trueonly = count_level_trueonly / lvl.shape[0]
#     return percentage_trueonly
#
# mouse_levels = levels_assigned(mouse_test_pred)
# rat_levels = levels_assigned(rat_pred)
# human_levels = levels_assigned(human_pred)
# zebrafish_levels = levels_assigned(zebrafish_pred)
# celegans_levels = levels_assigned(celegans_pred)
# yeast_levels = levels_assigned(yeast_pred)
# athaliana_levels = levels_assigned(athaliana_pred)
#
#
# print('true only predictions')
# print(mouse_levels)
# print(rat_levels)
# print(human_levels)
# print(zebrafish_levels)
# print(celegans_levels)
# print(yeast_levels)
# print(athaliana_levels)
#
# depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# depths = depth + depth + depth + depth + depth + depth + depth
# species = ['mouse_test','mouse_test','mouse_test','mouse_test','mouse_test','mouse_test','mouse_test','mouse_test','mouse_test','mouse_test','mouse_test',
#            'rat', 'rat', 'rat', 'rat', 'rat', 'rat', 'rat', 'rat', 'rat', 'rat', 'rat',
#            'human', 'human', 'human', 'human', 'human', 'human', 'human', 'human', 'human', 'human', 'human',
#            'zebrafish','zebrafish','zebrafish','zebrafish','zebrafish','zebrafish','zebrafish','zebrafish','zebrafish','zebrafish','zebrafish',
#            'celegans','celegans','celegans','celegans','celegans','celegans','celegans','celegans','celegans','celegans','celegans',
#            'yeast', 'yeast', 'yeast', 'yeast', 'yeast', 'yeast', 'yeast', 'yeast', 'yeast', 'yeast', 'yeast',
#            'athaliana', 'athaliana','athaliana','athaliana','athaliana','athaliana','athaliana','athaliana','athaliana','athaliana','athaliana']
# percentages = np.concatenate([mouse_levels, rat_levels, human_levels, zebrafish_levels, celegans_levels, yeast_levels, athaliana_levels])
#
#
# df_in_all = pd.DataFrame({'Species': species, 'Depth': depths, 'Percentage': percentages})
#
# now = datetime.now()
# current_time = now.strftime("%d%m%Y%H%M%S")
# plt.figure()
# sns.set(style="whitegrid")
# sns.swarmplot(y="Percentage", x="Depth", hue='Species', data=df_in_all, size=6)
# plt.title('across species function performance')
# plt.tight_layout()
# plt.savefig('figures/percentageTP_species' + current_time + '.png')
# plt.savefig('figures/percentageTP_species' + current_time + '.pdf')
# plt.close()
#
#
#
#
#











# # #
# # with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GO_terms/GO_depth_all_mouse_proteins.pkl', 'rb') as fw:
# #     depth_GO = pickle.load(fw)
# #
# # with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/mouse_valid.pkl', 'rb') as f:
# #     mousevalid_indexes = pickle.load(f)
# # with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/mouse.pkl', 'rb') as f:
# #     mouse_index = pickle.load(f)
# #
# # def GO_levels(index):
# #     levels = np.zeros((len(index),))
# #     for i, x in enumerate(index):
# #         levels[i] = int(depth_GO[x][0])
# #     return {'depth': levels}
# #
# # mouse_train_pred = GO_levels(mouse_index)
# # mouse_valid_pred = GO_levels(mousevalid_indexes)
# #
# # N = 11
# # ind = np.arange(N)    # the x locations for the groups
# # width = 0.35
# # labels = ['mouse train', 'mouse valid', 'mouse test', 'rat', 'human', 'zebrafish', 'c. elegans',
# #           'yeast', 'a. thaliana']
# #
# # def prep_bar(pred):
# #     count_levels = np.zeros(11)
# #     for value in pred['depth']:
# #         if int(value) >= 11:
# #             value = 11
# #         count_levels[int(value)-1] = count_levels[int(value)-1] + 1
# #     count_levels = count_levels / len(pred['depth'])
# #     bottoms = np.zeros(11)
# #     for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
# #         bottoms[i] = np.sum(count_levels[0:i])
# #     return count_levels, bottoms
# #
# # all_levels = np.zeros((9, 11))
# # all_levels[0, :] = prep_bar(mouse_train_pred)[0]
# # all_levels[1, :] = prep_bar(mouse_valid_pred)[0]
# # all_levels[2, :] = prep_bar(mouse_test_pred)[0]
# # all_levels[3, :] = prep_bar(rat_pred)[0]
# # all_levels[4, :] = prep_bar(human_pred)[0]
# # all_levels[5, :] = prep_bar(zebrafish_pred)[0]
# # all_levels[6, :] = prep_bar(celegans_pred)[0]
# # all_levels[7, :] = prep_bar(yeast_pred)[0]
# # all_levels[8, :] = prep_bar(athaliana_pred)[0]
# # all_bottoms = np.zeros((9, 11))
# # all_bottoms[0, :] = prep_bar(mouse_train_pred)[1]
# # all_bottoms[1, :] = prep_bar(mouse_valid_pred)[1]
# # all_bottoms[2, :] = prep_bar(mouse_test_pred)[1]
# # all_bottoms[3, :] = prep_bar(rat_pred)[1]
# # all_bottoms[4, :] = prep_bar(human_pred)[1]
# # all_bottoms[5, :] = prep_bar(zebrafish_pred)[1]
# # all_bottoms[6, :] = prep_bar(celegans_pred)[1]
# # all_bottoms[7, :] = prep_bar(yeast_pred)[1]
# # all_bottoms[8, :] = prep_bar(athaliana_pred)[1]
# #
# # now = datetime.now()
# # current_time = now.strftime("%d%m%Y%H%M%S")
# # fig, ax = plt.subplots()
# #
# # ax.bar(labels, all_levels[:, 0], width, label='Depth 1')
# # for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
# #     ax.bar(labels, all_levels[:, i], width, bottom=all_bottoms[:, i], label='Depth %s' % str(i+1))
# #
# # plt.gcf().subplots_adjust(bottom=0.20)
# # plt.gcf().subplots_adjust(right=0.8)
# # plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
# # plt.xticks(rotation=45)
# # plt.grid(axis='y', alpha=0.75)
# # plt.ylabel('Fraction of total tested terms')
# # plt.title('Distribution of depth GO terms')
# # plt.savefig('stacked_barplot_depthGO' + current_time + '.png')
# # plt.savefig('stacked_barplot_depthGO' + current_time + '.pdf')
# # plt.close()
