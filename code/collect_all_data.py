# #########################################################################
# Collect data from multiple analyses in a single file
# @MP Coll, 2020, michelpcoll@gmail.com
# #########################################################################

import pandas as pd
from os.path import join as opj
from bids import BIDSLayout
import numpy as np
###############################
# Parameters
###############################
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Remove stupid pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

param = {
         # excluded participants
         'excluded': ['sub-24', 'sub-31', 'sub-35', 'sub-51'],

         }
# Remove excluded part
part = [p for p in part if p not in param['excluded']]

# Load Model data

# Winning model
mod = 'HGF2_intercue'
comp_data = pd.read_csv(opj('/data/derivatives', 'computational_models',
                            mod, mod + '_data.csv'))

comp_data['sub'] = ['sub-' + str(p) for p in comp_data['sub']]

# Add NFR and ratings
comp_data['nfr_auc'] = 'nan'
comp_data['nfr_auc_z'] = 'nan'
comp_data['ratings'] = 'nan'
comp_data['ratings_z'] = 'nan'
subdats = []
for p in part:
    # Add NFR
    nfr = pd.read_csv(opj('/data/derivatives', p, 'emg',
                          p + '_task-fearcond_nfrauc.csv'))

    subdat = comp_data[comp_data['sub'] == p]

    nfr_auc = np.asarray(subdat['nfr_auc'])
    nfr_auc[subdat.cond == 'CS++'] = list(list(nfr['nfr_auc']))
    subdat['nfr_auc'] = nfr_auc

    nfr_auc_z = np.asarray(subdat['nfr_auc_z'])
    nfr_auc_z[subdat.cond == 'CS++'] = list(list(nfr['nfr_auc_z']))
    subdat['nfr_auc_z'] = nfr_auc_z

    ratings = np.asarray(subdat['ratings'])
    ratings[subdat.cond == 'CS++'] = list(list(nfr['ratings']))
    subdat['ratings'] = ratings

    ratings_z = np.asarray(subdat['ratings_z'])
    ratings_z[subdat.cond == 'CS++'] = list(list(nfr['ratings_z']))
    subdat['ratings_z'] = ratings_z

    subdats.append(subdat)

comp_data = pd.concat(subdats)

# Add erps
erps_meta = pd.read_csv('/data/derivatives/task-fearcond_erpsmeta.csv')

# Remove excluded subs
erps_meta = erps_meta[erps_meta['participant_id'].isin(list(comp_data['sub']))]

if not list(erps_meta['participant_id']) == list(comp_data['sub']):
    raise 'df not sorted'

comp_data = pd.concat([comp_data.reset_index(), erps_meta.reset_index()],
                      axis=1)

# Add questionnaires and socio
quest = pd.read_csv('/data/source/participants.tsv', sep='\t')

subdats = []
for p in part:
    # Add NFR
    subdat = comp_data[comp_data['sub'] == p]
    questsub = quest[quest['participant_id'] == p].reset_index()

    for c in questsub.columns.values:
        subdat[c] = questsub[c].values[0]

    subdats.append(subdat)

alldata = pd.concat(subdats)

alldata.to_csv('/data/derivatives/task-fearcond_alldata.csv')
