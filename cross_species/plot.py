import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# plot the results
# included values are for the Cellular components (CC)

### ROCAUC ###
labels = ['Mouse test', 'Rat', 'Human', 'Zebrafish', 'C. Elegans', 'Yeast', 'A. Thaliana']
MLP = np.array([0.853, 0.938, 0.930, 0.927, 0.894, 0.846, 0.897])
psiBLAST = np.array([0.720, 0.909, 0.888, 0.895, 0.832, 0.728, 0.795])


yer_test = np.array([[0.852, 0.855], [0.938, 0.939], [0.929, 0.930],
                     [0.925, 0.929], [0.893, 0.896], [0.844, 0.850],
                     [0.896, 0.899]])  # 95% confidence interval
yerr_test = np.c_[MLP - yer_test[:, 0], yer_test[:, 1] - MLP].T

yer_psiblast = np.array([[0.718, 0.722], [0.908, 0.910], [0.887, 0.889],
                     [0.891, 0.899], [0.829, 0.835], [0.724, 0.731],
                     [0.793, 0.798]])  # 95% confidence interval
yerr_psiblast = np.c_[psiBLAST - yer_psiblast[:, 0], yer_psiblast[:, 1] - psiBLAST].T

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
fig, ax = plt.subplots()
rects1 = ax.bar(x + (1/2)*width, MLP, width, label='MLP', color='darkorange', yerr=yerr_test, capsize=7)
rects2 = ax.bar(x - (1/2)*width, psiBLAST, width, label='psiBLAST', color='saddlebrown', yerr=yerr_psiblast, capsize=7)


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('ROCAUC-score')
ax.set_title('ROCAUC-scores by species - cellular component')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 6),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.savefig('figures/rocauc_performance' + current_time + '.png')
plt.savefig('figures/rocauc_performance' + current_time + '.pdf')
plt.close()



 


### fMAX ###
labels = ['Mouse test', 'Rat', 'Human', 'Zebrafish', 'C. Elegans', 'Yeast', 'A. Thaliana']
MLP = np.array([0.633, 0.670, 0.647, 0.636, 0.592, 0.621, 0.643])
psiBLAST = np.array([0.473, 0.676, 0.614, 0.623, 0.450, 0.372, 0.392])

yer_test = np.array([[0.631, 0.636], [0.669, 0.672], [0.645, 0.648], [0.631, 0.640], [0.590, 0.595], [0.619, 0.624], [0.641, 0.645]])  # 95% confidence interval
yerr_test = np.c_[MLP - yer_test[:, 0], yer_test[:, 1] - MLP].T

yer_psiblast = np.array([[0.467, 0.477], [0.673, 0.679], [0.612, 0.616], [0.616, 0.630], [0.443, 0.456], [0.367, 0.378], [0.389, 0.396]])  # 95% confidence interval
yerr_psiblast = np.c_[psiBLAST - yer_psiblast[:, 0], yer_psiblast[:, 1] - psiBLAST].T


x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")
fig, ax = plt.subplots()
rects1 = ax.bar(x + (1/2)*width, MLP, width, label='MLP', color='darkorange', yerr=yerr_test, capsize=7)
rects2 = ax.bar(x - (1/2)*width, psiBLAST, width, label='psiBLAST', color='saddlebrown', yerr=yerr_psiblast, capsize=7)


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1-score')
ax.set_title('F1-scores by species - cellular component')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 6),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def autolabelinbar(rects, cov):
    """Attach a text label above each bar in *rects*, displaying its height.""" 
    coverage = list('C = %s' % x for x in cov)
    coverage = np.array(coverage)
    for i, rect in enumerate(rects):
        ax.annotate(coverage[i],
                    xy=(rect.get_x() + rect.get_width() / 2, 0),
                    xytext=(0, 5),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation='vertical')

cov_mlp = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
cov_psiblast = [0.74, 0.92, 0.89, 0.88, 0.64, 0.52, 0.56]

autolabel(rects1)
autolabelinbar(rects1, cov_mlp)
autolabel(rects2)
autolabelinbar(rects2, cov_psiblast)
fig.tight_layout()
plt.savefig('figures/fmax_performance' + current_time + '.png')
plt.savefig('figures/fmax_performance' + current_time + '.pdf')
plt.close()
