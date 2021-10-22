import sys
import collections as cx
import numpy as np
import pickle

# with open('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/data/termNames_seq_filtered.pkl', 'rb') as f:
#     term = pickle.load(f)
#
# terms2keep = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/ameliavm/sequence-only/lists/termIndicesToUse.npy')

directory = '/tudelft.net/staff-bulk/ewi/insy/DBL/ivandenbent/final/cross_species/embeddings/cellular_component/evidence_codes'

type_GO = 'cellular_component'

from goatools.obo_parser import GODag
obodag = GODag("go-basic.obo")

class RptLevDepth(object):
    """Reports a summary of GO term levels in depths."""

    nss = ['biological_process', 'molecular_function', 'cellular_component']

    def __init__(self, obodag, log=sys.stdout):
        self.obo = obodag
        self.log = log
        self.title = "GO Counts in {VER}".format(VER=self.obo.version)

    def write_cnts(self, go_ids):
        """Write summary of level and depth counts for specific GO ids."""
        obo = self.obo
        cnts = self.get_cnts_levels_depths_recs([obo.get(GO) for GO in go_ids])
        yu = []
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            if cnts['level'][i][type_GO] ==1:
                yu.append(i)
        return yu


    @staticmethod
    def get_cnts_levels_depths_recs(recs):
        """Collect counts of levels and depths in a Group of GO Terms."""
        cnts = cx.defaultdict(lambda: cx.defaultdict(cx.Counter))
        for rec in recs:
            if rec is not None and not rec.is_obsolete:
                cnts['level'][rec.level][rec.namespace] += 1
                cnts['depth'][rec.depth][rec.namespace] += 1
        return cnts

    def get_data(self):
        """Collect counts of GO terms at all levels and depths."""
        # Count level(shortest path to root) and depth(longest path to root)
        # values for all unique GO Terms.
        data = []
        ntobj = cx.namedtuple("NtGoCnt", "Depth_Level BP_D MF_D CC_D BP_L MF_L CC_L")
        cnts = self.get_cnts_levels_depths_recs(set(self.obo.values()))
        max_val = max(max(dep for dep in cnts['depth']), max(lev for lev in cnts['level']))
        for i in range(max_val+1):
            vals = [i] + [cnts[desc][i][ns] for desc in cnts for ns in self.nss]
            data.append(ntobj._make(vals))
        return data

with open('%s/unique_per_species.pkl' % directory, 'rb') as joe:
    unique = pickle.load(joe)


rptobj = RptLevDepth(obodag)
level_GO = {}
for k in unique['mouse']:
    level_GO[k] = rptobj.write_cnts([k])

with open('%s/GO_levels_all_GOterms.pkl' % directory, 'wb') as fw:
    pickle.dump(level_GO, fw)
