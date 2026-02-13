import pandas as pd

import load
import matplotlib.pyplot as plt
import setup
import paths

save_path = paths.save_path
plot_path = paths.plots_path
exp_info = setup.exp_info()

#----- Save data and display figures -----#
save_data = True
save_fig = True
display_figs = True
use_saved_data = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

meg_params = {'chs_id': 'mag',
              'band_id': None,
              'filter_sensors': True,
              'filter_method': 'iir',
              'data_type': 'ICA_annot',  # 'raw', 'ICA', 'processed', 'tsss'
              }

all_sac = []

for subject_id in exp_info.subjects_ids:

    # Load subject
    subject = setup.subject(subject_id=subject_id)

    # Load MEG data
    # meg_data = load.meg(subject_id=subject_id, meg_params=meg_params)

    # Load saccades
    saccades = subject.saccades()

    all_sac.append(saccades)

ga_sac = pd.concat(all_sac)

print(ga_sac['duration'].median())
