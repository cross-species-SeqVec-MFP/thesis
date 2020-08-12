import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils import resample 


#this data is scaled already
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/data/XYdata_scaled_train', 'rb') as fw:
    Xtrain, Ytrain, GO_terms = pickle.load(fw)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/data/XYdata_scaled_valid', 'rb') as fw:
    Xvalid, Yvalid, GO_terms = pickle.load(fw)
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/data/XYdata_scaled_test', 'rb') as fw:
    Xtest, Ytest, GO_terms = pickle.load(fw)

location = {}
for z, term in enumerate(GO_terms):
    location[term] = z

i = int(sys.argv[1]) #batchloop over i in 0 1 2 3

if i == 3:
    #get summed amino acid level embeddings
    sum1 = np.add(Xtrain[:, 0:1024], Xtrain[:, 1024:2048])
    Xtrain = np.add(sum1, Xtrain[:, 2048:3072])
    sum2 = np.add(Xvalid[:, 0:1024], Xvalid[:, 1024:2048])
    Xvalid = np.add(sum2, Xvalid[:, 2048:3072])
    sum3 = np.add(Xtest[:, 0:1024], Xtest[:, 1024:2048])
    Xtest = np.add(sum3, Xtest[:, 2048:3072])

    start_index = 0
    end_index = 1024
    print('Start index: %s' % start_index)
    print('End index: %s' % end_index)

else:
    start_index = np.array([0, 1024, 2048])[i] #0, 1024 or 2048
    end_index = np.array([3072, 2048, 3072])[i] #2048 or 3072
 
    Xtrain = Xtrain[:, start_index:end_index]
    Xvalid = Xvalid[:, start_index:end_index]
    Xtest = Xtest[:, start_index:end_index]

    print('Start index: %s' % start_index)
    print('End index: %s' % end_index)

protein_id_train = np.loadtxt('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/lists/train_final.names', dtype=str).tolist()
protein_id_valid = np.loadtxt('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/lists/valid_final.names', dtype=str).tolist()
protein_id_test = np.loadtxt('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/lists/test_final.names', dtype=str).tolist()

index_train = np.arange(0, len(protein_id_train))
index_valid = np.arange(0, len(protein_id_valid))
index_test = np.arange(0, len(protein_id_test))


def fmax(Ytrue, Ypost1):
    thresholds = np.linspace(0.0, 1.0, 51)
    ff = np.zeros(thresholds.shape, dtype=object)
    pr = np.zeros(thresholds.shape, dtype=object)
    rc = np.zeros(thresholds.shape, dtype=object)
    rc_avg = np.zeros(thresholds.shape)
    pr_avg = np.zeros(thresholds.shape)
    coverage = np.zeros(thresholds.shape)

    Ytrue = Ytrue.transpose()
    Ypost1 = Ypost1.transpose()
    tokeep = np.where(np.sum(Ytrue, 0) > 0)[0]
    Ytrue = Ytrue[:, tokeep]
    Ypost1 = Ypost1[:, tokeep]

    for i, t in enumerate(thresholds):
        _ , rc[i], _ , _ = precision_recall_fscore_support(Ytrue, (Ypost1 >= t).astype(int))
        rc_avg[i] = np.mean(rc[i])

        tokeep = np.where(np.sum((Ypost1 >= t).astype(int), 0) > 0)[0]
        Ytrue_pr = Ytrue[:, tokeep]
        Ypost1_pr = Ypost1[:, tokeep]
        if tokeep.any():
            pr[i], _, _, _ = precision_recall_fscore_support(Ytrue_pr, (Ypost1_pr >= t).astype(int))
            pr_avg[i] = np.mean(pr[i])
            coverage[i] = len(pr[i])/len(rc[i])

        ff[i] = (2 * pr_avg[i] * rc_avg[i]) / (pr_avg[i] + rc_avg[i])

    return np.nanmax(ff), coverage[np.argmax(ff)], thresholds[np.argmax(ff)]

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
N, D_in, H, D_out = 64, Xtrain.shape[1], 512, Ytrain.shape[1]

Xtrain = torch.from_numpy(Xtrain)
Ytrain = torch.from_numpy(Ytrain)
index_train = torch.from_numpy(index_train)

Xvalid = torch.from_numpy(Xvalid)
Yvalid = torch.from_numpy(Yvalid)
index_valid = torch.from_numpy(index_valid)

Xtest = torch.from_numpy(Xtest)
Ytest = torch.from_numpy(Ytest)
index_test = torch.from_numpy(index_test)

dataset_train = TensorDataset(Xtrain, Ytrain, index_train)
dataset_valid = TensorDataset(Xvalid, Yvalid, index_valid)
dataset_test = TensorDataset(Xtest, Ytest, index_test)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create random Tensors to hold inputs and outputs
train_loader = DataLoader(dataset=dataset_train, batch_size=N, shuffle=True)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=N, shuffle=True)
test_loader = DataLoader(dataset=dataset_test, batch_size=N, shuffle=True)


# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output.

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

# optimizer functions
# want to prevent overfitting (no perfect weights on train set)
learning_rate = 5e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define scheduler for learning rate adjustment
# don't want to get stuck in 'local minimum'
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
 
# define loss function loss functions;
# contains the sigmoid function at the end
loss_fn = torch.nn.MultiLabelSoftMarginLoss()

rocauc_scores = []
fmax_scores = []
coverage_scores = []
threshold_scores = []
epochie = []
epochs = 75
for epoch in range(epochs):
    model.train()
    for data in train_loader:
        # data is batch of features and labels
        X, y, index = data

        # Zero the gradients before running the backward pass.
        model.zero_grad()
        optimizer.zero_grad()

        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model.forward(X.float())

        # Compute loss.
        loss = loss_fn(y_pred, y)

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model.
        loss.backward()

        # optimizer to adjust weights
        optimizer.step()

    # finding out how well the model is working on validation
    model.eval()

    avg_valid_loss = 0
    y_true = []
    y_pred_sigm = []
    with torch.no_grad():
        for data in valid_loader:
            X, y, index = data
            output = model.forward(X.float())
            valid_loss = loss_fn(output, y)
            avg_valid_loss += valid_loss.item() / len(train_loader)
            y_true.append(y.cpu().numpy().squeeze())
            y_pred_sigm.append(torch.sigmoid(output).cpu().numpy().squeeze())

    # adjust learning rate
    scheduler.step(avg_valid_loss)

    # Calculate evaluation metrics
    y_true1 = np.vstack(y_true)
    y_pred_sigm1 = np.vstack(y_pred_sigm)

    # ROC AUC score
    ii = np.where(np.sum(y_true1, 0) > 0)[0]
    avg_rocauc = np.nanmean(roc_auc_score(y_true1[:, ii], y_pred_sigm1[:, ii], average=None))
    rocauc_scores.append(avg_rocauc)
    epochie.append(epoch)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'learning_rate': learning_rate,
        'Ytrue': y_true1,
        'Ypred': y_pred_sigm1,
    }, '/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/layers/Trained_epochs/epoch%s_fea%s:%s' % (epoch, start_index, end_index))


with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/layers/Trained_epochs/valid_results_fea%s:%s' % (start_index, end_index), 'wb') as f:
    pickle.dump([rocauc_scores, epochie], f)

rocauc_scores = np.array(rocauc_scores)
print('best model for epoch: %s' % np.argmax(rocauc_scores))

######################### test set
model1 = MLP()

with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/data/Seeds_random_state', 'rb') as f:
    seeds = pickle.load(f)

# for correction GO hierarchy
with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/data/GO_depth_all_proteins.pkl', 'rb') as fw:
    depth_terms = pickle.load(fw)

max_value = 0
terms_per_depth = {k: [] for k in range(15)}
for key in location.keys():
    depth = depth_terms[key][0]
    terms_per_depth[depth].append(key)
    if depth > max_value:
        max_value = depth


best_epoch = np.argmax(rocauc_scores)

checkpoint1 = torch.load('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/layers/Trained_epochs/epoch%s_fea%s:%s' % (best_epoch, start_index, end_index))
model1.load_state_dict(checkpoint1['model_state_dict'])

model1.eval()


def predictions_term(loader_):
    y_true = []
    y_pred_sigm = []
    index_t = []
    with torch.no_grad():
        for data in loader_:
            X, y, index = data
            output = model1.forward(X.float())
            y_true.append(y.cpu().numpy().squeeze())
            y_pred_sigm.append(torch.sigmoid(output).cpu().numpy().squeeze())
            index_t.append(index.cpu().numpy().squeeze())

    # Calculate evaluation metrics
    y_true1 = np.vstack(y_true)
    y_pred_sigm1 = np.vstack(y_pred_sigm)
    index_tt = index_t

    # correcting for GO hierarchy
    for depth1 in np.arange(max_value, -1, -1):
        for key11 in GO_terms:
            if key11 in terms_per_depth[depth1]:
                if depth_terms[key11][1]:
                    for up_terms in depth_terms[key11][1]:
                        if up_terms in GO_terms:
                            index1 = np.where(y_pred_sigm1[:, location[key11]] < y_pred_sigm1[:, location[up_terms]])[0]
                            if index1.any():
                                for protein in index1:
                                    y_pred_sigm1[protein, location[key11]] = y_pred_sigm1[protein, location[up_terms]]

    # ROC AUC score
    iii = np.where(np.sum(y_true1, 0) > 0)[0]
    avg_rocauc = np.nanmean(roc_auc_score(y_true1[:, iii], y_pred_sigm1[:, iii], average=None))

    print('rocauc: %s' % avg_rocauc)

    boot_results_rocauc = np.zeros(len(seeds))

    for ij, seed in enumerate(seeds):
        Ypost, Yval = resample(y_pred_sigm1, y_true1, random_state=seed, stratify=y_true1)
        iiii = np.where(np.sum(Yval, 0) > 0)[0]
        boot_results_rocauc[ij] = np.nanmean(roc_auc_score(Yval[:, iiii], Ypost[:, iiii], average=None))

    print('confidence interval')
    print(np.percentile(boot_results_rocauc, [2.5, 97.5]))
    sys.stdout.flush()

    # fmax score
    avg_fmax, cov, threshold = fmax(y_true1, y_pred_sigm1)

    print('f1-score: %s' % avg_fmax)
    print('coverage: %s' % cov)
    print('threshold: %s' % threshold)

    boot_results_f1 = np.zeros(len(seeds))

    for ji, seed in enumerate(seeds):
        Ypost, Yval = resample(y_pred_sigm1, y_true1, random_state=seed, stratify=y_true1)
        fmea = fmax_threshold(Yval, Ypost, threshold)
        boot_results_f1[ji] = fmea[0]

    print('confidence interval')
    print(np.percentile(boot_results_f1, [2.5, 97.5]))
    sys.stdout.flush()


    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/layers/Predictions/predictions_term_centric_NNfea%s:%s.pkl' % (start_index, end_index), 'wb') as fw:
        pickle.dump({'Yval': y_true1, 'Ypost': y_pred_sigm1, 'indexes': index_tt, 'boot_rocauc': boot_results_rocauc, 'boot_f1': boot_results_f1}, fw)


predictions_term(test_loader)

print('done')









