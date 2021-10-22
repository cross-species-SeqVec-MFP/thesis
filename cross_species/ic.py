with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/GO_terms/GO_depth_all_mouse_proteins.pkl', 'rb') as fw:
    depth_terms = pickle.load(fw)

# calculate information content values
def IC_val(y, loc, species):
    num = np.sum(y, axis=0)
    tot = np.sum(y)

    IC = {}
    tot_IC = 0
    for term in loc.keys():
        IC[term] = -np.log(num[loc[term]] / tot)
        tot_IC = tot_IC + -np.log(num[loc[term]] / tot)

    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_term_centric/' + species + '.pkl','rb') as f:
        term_index = pickle.load(f)
    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/' + species + '.pkl','rb') as f:
        protein_index = pickle.load(f)

    tot_prot = 0
    for i in protein_index:
        tot_prot = tot_prot + IC[i]

    ic = []
    de = []
    for j in loc.keys():
        if j not in protein_index:
            ic.append(IC[j])
            de.append(depth_terms[j][0])
    ic = np.array(ic)

    print(species)
    print(np.mean(ic))
    print(np.std(ic))

    tot_term = 0
    for ii in term_index:
        tot_term= tot_term + IC[ii]

    return tot_IC, tot_prot/tot_IC, tot_term/tot_IC

print(IC_val(yrat, loc_rat, 'rat'))
print(IC_val(yhuman, loc_human, 'human'))
print(IC_val(yzebrafish, loc_zebrafish, 'zebrafish'))
print(IC_val(ycelegans, loc_celegans, 'celegans'))
print(IC_val(yyeast, loc_yeast, 'yeast'))
print(IC_val(yathaliana, loc_athaliana, 'athaliana'))

#for the mouse sets
def IC_val(y, loc, species, indexes):
    num = np.sum(y, axis=0)
    tot = np.sum(y)

    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_term_centric/' + species + '.pkl','rb') as f:
        term_index = pickle.load(f)
    with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/Mouse_model/Unique_GO_terms/index_protein_centric/' + species + '.pkl','rb') as f:
        protein_index = pickle.load(f)

    IC = {}
    tot_IC = 0
    for term in loc.keys():
        if num[loc[term]] != 0:
            IC[term] = -np.log(num[loc[term]] / tot)
            tot_IC = tot_IC + -np.log(num[loc[term]] / tot)

    tot_prot = 0
    for i in protein_index:
        tot_prot = tot_prot + IC[i]

    ic = []
    for j in indexes:
        if j not in protein_index:
            ic.append(IC[j])
    ic = np.array(ic)

    print(species)
    print(np.mean(ic))
    print(np.std(ic))

    tot_term = 0
    for ii in term_index:
        tot_term= tot_term + IC[ii]

    return tot_IC, tot_prot/tot_IC, tot_term/tot_IC

print(IC_val(ymouse_valid, loc_mouse, 'mouse_valid', mousevalid_indexes))
print(IC_val(ymouse_test, loc_mouse, 'mouse_test', mousetest_indexes))
