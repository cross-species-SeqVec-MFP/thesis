import numpy as np
import pickle
import sys

path = sys.argv[1]
path1 = sys.argv[2]

with open('%s/mouse_index_traintestvalid.pkl' % path, 'rb') as f:
    mouse_train_ind, mouse_validtest_ind, mouse_valid_ind, mouse_test_ind = pickle.load(f)

with open('%s/mouse.pkl' % path, 'rb') as f:
    ymouse, Xm, prot_mouse, loc_mouse = pickle.load(f)

prot_train = prot_mouse[mouse_train_ind]
prot_validtest = prot_mouse[mouse_validtest_ind]
prot_valid = prot_validtest[mouse_valid_ind]
prot_test = prot_validtest[mouse_test_ind]

# Dictionairy info contains protein id and sequence of protein of interest
# and the label for their class
boo = False
info = {}
seq = ""
with open('%s/mouse_proteinsequence.fasta' % path1, 'r') as f:
    lines = f.read().splitlines()
    last_line = lines[-1]
with open('%s/mouse_proteinsequence.fasta' % path1) as f:
    for line in f:
        if line[0] == '>':
            if boo:
                info[protein_id] = seq
                seq = ""
            boo = True
            protein_id = line.split('|')[1]
        else:
            seq = seq + line[0:-1]
        if line[0:-1] == last_line:
            info[protein_id] = seq


ofile = open("%s/mouse_train_proteinsequence.fasta" % path1, "w")
for i in prot_train:
    ofile.write(">" + i + "\n" + info[i] + "\n")
ofile.close()

ofile = open("%s/mouse_valid_proteinsequence.fasta" % path1, "w")
for i in prot_valid:
    ofile.write(">" + i + "\n" + info[i] + "\n")
ofile.close()

ofile = open("%s/mouse_test_proteinsequence.fasta" % path1, "w")
for i in prot_test:
    ofile.write(">" + i + "\n" + info[i] + "\n")
ofile.close()
