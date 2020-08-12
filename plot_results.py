import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


#####################
# performance embedding types for LR classifier

fig, ax = plt.subplots()
# layers baseline, lstm1, lstm2, concatenated
x = [1024, 1024, 1024, 3072]
y = [0.8323, 0.844, 0.822, 0.838]
e = np.abs([np.array([0.8265, 0.8371]) - 0.8323, np.array([0.839, 0.848]) - 0.844, np.array([0.818, 0.826]) - 0.822, np.array([0.834, 0.843]) - 0.838]).T
ax.errorbar(x, y, yerr=e, fmt='o', label='Layers')

# Baseline based generalized mean
x = [3072, 5120, 10240]
y = [0.8382, 0.8436, 0.839]
e = np.abs([np.array([0.8325, 0.8434]) - 0.8382, np.array([0.8378, 0.8482]) - 0.8436, np.array([0.835, 0.843]) - 0.839]).T
ax.errorbar(x, y, yerr=e, fmt='o', color='tomato', label='Baseline based generalized means')

# Baseline based moments
x = [3072, 5120, 10240]
y = [0.8376, 0.8396, 0.8415]
e = np.abs([np.array([0.8321, 0.8425]) - 0.8376, np.array([0.8344, 0.8445]) - 0.8396, np.array([0.8363, 0.8463]) - 0.8415]).T
ax.errorbar(x, y, yerr=e, fmt='o', color='darkred', label='Baseline based moments')

# lstm1 based generalized mean
x = [3072, 5120, 10240]
y = [0.848, 0.852, 0.857]
e = np.abs([np.array([0.844, 0.853]) - 0.848, np.array([0.848, 0.857]) - 0.852, np.array([0.853, 0.862]) - 0.857]).T
ax.errorbar(x, y, yerr=e, fmt='o', color='gold', label='biLSTM 1 based generalized means')

# Baseline based moments
x = [3072, 5120, 10240]
y = [0.848, 0.850, 0.851]
e = np.abs([np.array([0.844, 0.853]) - 0.848, np.array([0.846, 0.854]) - 0.850, np.array([0.847, 0.856]) - 0.851]).T
ax.errorbar(x, y, yerr=e, fmt='o', color='sandybrown', label='biLSTM 1 based moments')


now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average rocauc score')
ax.set_xlabel('Dimensionality')
ax.set_title('Rocauc scores by number and type of features')
ax.set_ylim(0.81, 0.87)
ax.legend()
plt.grid(alpha=0.75)
plt.savefig('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/Figures/performance_features_LR' +current_time +'.pdf')
plt.close()



#####################
# performance embedding types for MLP classifier

fig, ax = plt.subplots()
# layers baseline, lstm1, lstm2, concatenated
x = [1024, 1024, 1024, 3072]
y = [0.856, 0.857, 0.856, 0.860]
e = np.abs([np.array([0.851, 0.861]) - 0.856, np.array([0.851, 0.861]) - 0.857, np.array([0.851, 0.860]) - 0.856, np.array([0.856, 0.866]) - 0.860]).T
ax.errorbar(x, y, yerr=e, fmt='o', label='Layers')

# Baseline based generalized mean
x = [3072, 5120, 10240]
y = [0.854, 0.860, 0.843]
e = np.abs([np.array([0.850, 0.859]) - 0.854, np.array([0.854, 0.864]) - 0.860, np.array([0.838, 0.848]) - 0.843]).T
ax.errorbar(x, y, yerr=e, fmt='o', color='tomato', label='Baseline based generalized means')

# Baseline based moments
x = [3072, 5120, 10240]
y = [0.855, 0.851, 0.844]
e = np.abs([np.array([0.849, 0.859]) - 0.855, np.array([0.846, 0.856]) - 0.851, np.array([0.838, 0.848]) - 0.844]).T
ax.errorbar(x, y, yerr=e, fmt='o', color='darkred', label='Baseline based moments')

# lstm1 based generalized mean
x = [3072, 5120, 10240]
y = [0.855, 0.858, 0.849]
e = np.abs([np.array([0.851, 0.860]) - 0.855, np.array([0.853, 0.863]) - 0.858, np.array([0.843, 0.853]) - 0.849]).T
ax.errorbar(x, y, yerr=e, fmt='o', color='gold', label='biLSTM 1 based generalized means')

# Baseline based moments
x = [3072, 5120, 10240]
y = [0.856, 0.845, 0.840]
e = np.abs([np.array([0.851, 0.860]) - 0.856, np.array([0.838, 0.849]) - 0.845, np.array([0.835, 0.845]) - 0.840]).T
ax.errorbar(x, y, yerr=e, fmt='o', color='sandybrown', label='biLSTM 1 based moments')


now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average rocauc score')
ax.set_xlabel('Dimensionality')
ax.set_title('Rocauc scores by number and type of features')
ax.set_ylim(0.81, 0.87)
ax.legend()
plt.grid(alpha=0.75)
plt.savefig('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/Figures/performance_features_MLP' +current_time +'.pdf')
plt.close()








####################################
# validation performance rocauc
Cs = [1e-3, 1e-2, 1e-1, 1, 10, 100]

#layers
lstm2 = [0.898, 0.885, 0.879, 0.896, 0.883, 0.865]
concatenated = [0.898, 0.897, 0.890, 0.913, 0.901, 0.874]
baseline = [0.90, 0.89, 0.88, 0.90, 0.89, 0.87]
lstm1 = [0.901, 0.893, 0.889, 0.903, 0.892, 0.880]

#baseline based
moment3072 = [0.88, 0.87, 0.86, 0.91, 0.90, 0.86]
moment5120 = [0.87, 0.86, 0.86, 0.91, 0.90, 0.86]
moment10240 = [0.86, 0.86, 0.86, 0.92, 0.91, 0.87]

genmean3072 = [0.87, 0.85, 0.84, 0.91, 0.90, 0.84]
genmean5120 = [0.88, 0.88, 0.87, 0.92, 0.91, 0.86]
genmean10240 = [0.892, 0.894, 0.885, 0.927, 0.918, 0.881]

#lstm1 based
lmoment3072 = [0.886, 0.879, 0.872, 0.911, 0.902, 0.869]
lmoment5120 = [0.881, 0.877, 0.871, 0.915, 0.906, 0.873]
lmoment10240 = [0.871, 0.879, 0.876, 0.919, 0.910, 0.879]

lgenmean3072 = [0.885, 0.871, 0.859, 0.912, 0.904, 0.861]
lgenmean5120 = [0.896, 0.890, 0.879, 0.921, 0.910, 0.876]
lgenmean10240 = [0.903, 0.902, 0.893, 0.930, 0.915, 0.888]




now = datetime.now()
current_time = now.strftime("%d%m%Y%H%M%S")

fig = plt.figure(constrained_layout=True)

gs = GridSpec(5, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :])
ax3 = fig.add_subplot(gs[2, :])
ax4 = fig.add_subplot(gs[3, :])
ax5 = fig.add_subplot(gs[4, :])

ax1.plot(Cs, baseline, color = 'red')
ax1.plot(Cs, concatenated, color = 'brown')
ax1.plot(Cs, lstm1, color = 'orange')
ax1.plot(Cs, lstm2, color = 'blue')
ax1.set_xscale('log')
ax1.set_ylim(0.83, 0.94)
ax1.grid(True)

ax2.plot(Cs, lmoment3072, color = 'red')
ax2.plot(Cs, lmoment5120, color = 'orange')
ax2.plot(Cs, lmoment10240, color = 'brown')
# ax2.set_title('SeqVec layer')
ax2.set_xscale('log')
ax2.set_ylim(0.83, 0.94)
ax2.grid(True)

ax3.plot(Cs, moment3072, color = 'orange')
ax3.plot(Cs, moment5120, color = 'yellow')
ax3.plot(Cs, moment10240, color = 'brown')
# ax4.set_title('SeqVec moments')
ax3.set_xscale('log')
ax3.set_ylim(0.83, 0.94)
ax3.grid(True)

ax4.plot(Cs, lgenmean3072, color = 'orange')
ax4.plot(Cs, lgenmean5120, color = 'yellow')
ax4.plot(Cs, lgenmean10240, color = 'brown')
# ax3.set_title('SeqVec gen mean')
ax4.set_xscale('log')
ax4.set_ylim(0.83, 0.94)
ax4.grid(True)

ax5.plot(Cs, genmean3072, color = 'orange')
ax5.plot(Cs, genmean5120, color = 'yellow')
ax5.plot(Cs, genmean10240, color = 'brown')
# ax3.set_title('SeqVec gen mean')
ax5.set_xscale('log')
ax5.set_ylim(0.83, 0.94)
ax5.grid(True)

plt.savefig('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/improve_seqvec/Figures//validation_rocauc_allembeddings' +current_time +'.pdf')
plt.close()

