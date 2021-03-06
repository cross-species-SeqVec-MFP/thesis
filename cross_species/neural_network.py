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

# this code is used to train the MLP on the mouse training set

directory = '/somedirectory/fasta_GO_embeddings'

directory1 = '/somedirectory/evidence_codes'

root_term = 'GO:0005575' #change here to root term of the BP, CC or MF

directory_epochs = '/somedirectory/neural_model/epochs'

with open('%s/mouse.pkl' % directory, 'rb') as f:
    ymouse, Xm, prot_mouse, loc_mouse = pickle.load(f)
with open('%s/mouse_index_setsX.pkl' % directory, 'rb') as f:
    Xmouse_train, Xmouse_valid, Xmouse_test = pickle.load(f)
with open('%s/mouse_index_setsY.pkl' % directory, 'rb') as f:
    ymouse_train, ymouse_valid, ymouse_test = pickle.load(f)
Xmouse_train = Xmouse_train[:, 0:1024]
Xmouse_valid = Xmouse_valid[:, 0:1024]

scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xmouse_train)
with open('%s/standard_scaler_trained' % directory1, 'wb') as f:
    pickle.dump(scaler, f)

Xvalid = scaler.transform(Xmouse_valid)

with open('%s/index_term_centric/mouse_valid.pkl' % directory1, 'rb') as f:
    mousevalid_indexes_term = pickle.load(f)

with open('%s/index_protein_centric/mouse_valid.pkl' % directory1, 'rb') as f:
    mousevalid_indexes_protein = pickle.load(f)

mousevalid_indexes_protein.remove(root_term)
mousevalid_indexes_term.remove(root_term)




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


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1024, 512, ymouse_train.shape[1]

Xtrain = torch.from_numpy(Xtrain)
ytrain = torch.from_numpy(ymouse_train)

Xvalid = torch.from_numpy(Xvalid)
yvalid = torch.from_numpy(ymouse_valid)

dataset_train = TensorDataset(Xtrain, ytrain)
dataset_valid = TensorDataset(Xvalid, yvalid)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create random Tensors to hold inputs and outputs
train_loader = DataLoader(dataset=dataset_train, batch_size=N, shuffle=True)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=N, shuffle=True)


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
epochs = 100
for epoch in range(epochs):
    model.train()
    for data in train_loader:
        # data is batch of features and labels
        X, y = data

        # Zero the gradients before running the backward pass.
        model.zero_grad()
        optimizer.zero_grad()

        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model.forward(X.float())

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        if epoch % 5 == 2:
            print(epoch, loss.item())

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
            X, y = data
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

    #get the terms for ROCAUC performance test
    y_true_term = np.zeros((y_true1.shape[0], len(mousevalid_indexes_term)))
    y_pred_sigm_term = np.zeros((y_true1.shape[0], len(mousevalid_indexes_term)))
    for ii, key in enumerate(mousevalid_indexes_term):
        y_true_term[:, ii] = y_true1[:, loc_mouse[key]]
        y_pred_sigm_term[:, ii] = y_pred_sigm1[:, loc_mouse[key]]

    # ROC AUC score
    ii = np.where(np.sum(y_true_term, 0) > 0)[0]
    avg_rocauc = np.nanmean(roc_auc_score(y_true_term[:, ii], y_pred_sigm_term[:, ii], average=None))


    #get the terms for fmax performance test
    y_true_protein = np.zeros((y_true1.shape[0], len(mousevalid_indexes_protein)))
    y_pred_sigm_protein = np.zeros((y_true1.shape[0], len(mousevalid_indexes_protein)))
    for i, key1 in enumerate(mousevalid_indexes_protein):
        y_true_protein[:, i] = y_true1[:, loc_mouse[key1]]
        y_pred_sigm_protein[:, i] = y_pred_sigm1[:, loc_mouse[key1]]

    # fmax score
    avg_fmax = fmax(y_true_protein, y_pred_sigm_protein)

    fmax_scores.append(avg_fmax[0])
    coverage_scores.append(avg_fmax[1])
    threshold_scores.append(avg_fmax[2])
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
    }, '%s/epoch%s' % (directory_epochs, epoch))

with open('%s/valid_results' % directory_epochs, 'wb') as f:
    pickle.dump([rocauc_scores, epochie, fmax_scores, coverage_scores, threshold_scores], f)


