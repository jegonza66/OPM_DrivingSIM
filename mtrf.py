import functions_analysis
import functions_general
import load
import setup
import paths
import matplotlib.pyplot as plt
import os

#----- Path -----#
exp_info = setup.exp_info()

#----- Save data and display figures -----#
use_saved_data = True
save_data = True
save_fig = True
display_figs = False
plot_individuals = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Parameters -----#
trial_params = {
                'evt_from_df': True,
                }

meg_params = {'chs_id': 'mag',
              'band_id': None,
              'data_type': 'ICA_annot'
              }

# TRF parameters
trf_params = {'input_features': {'fix': None,
                                 'sac': None
                                 },   # Select features (events)
              'standarize': False,
              'fit_power': False,
              'alpha': None,
              'tmin': -0.2,
              'tmax': 0.5,
              }
trf_params['baseline'] = (trf_params['tmin'], -0.05)

# Figure path
fig_path = paths.plots_path + (f"TRF_{meg_params['data_type']}/Band_{meg_params['band_id']}/{trf_params['input_features']}"
                               f"_{trf_params['tmin']}_{trf_params['tmax']}_bline{trf_params['baseline']}_alpha{trf_params['alpha']}_"
                               f"std{trf_params['standarize']}/{meg_params['chs_id']}/").replace(":", "")

# Change path to include envelope power
if trf_params['fit_power']:
    fig_path = fig_path.replace(f"TRF_{meg_params['data_type']}", f"TRF_{meg_params['data_type']}_ENV")

# Save path
save_path = fig_path.replace(paths.plots_path, paths.save_path)

# Define Grand average variables
feature_evokeds = {}

elements = trf_params['input_features'].keys()
for feature in elements:
    feature_evokeds[feature] = []
    if isinstance(trf_params['input_features'], dict):
        try:
            for value in trf_params['input_features'][feature]:
                feature_value = f'{feature}-{value}'
                feature_evokeds[feature_value] = []
        except:
            pass

# Iterate over subjects
for sub_idx, subject_id in enumerate(exp_info.subjects_ids):
    trf_path = save_path
    trf_fname = f'TRF_{subject_id}.pkl'

    # Load subject
    subject = setup.subject(subject_id=subject_id)
    # Load MEG data
    meg_data = load.meg(subject_id=subject_id, meg_params=meg_params)

    # Pick channels
    picks = functions_general.pick_chs(chs_id=meg_params['chs_id'], info=meg_data.info)
    meg_data = meg_data.pick(picks)

    if os.path.exists(trf_path + trf_fname) and use_saved_data:
        # Load TRF
        rf = load.var(trf_path + trf_fname)
        print('Loaded Receptive Field')

    else:
        # Compute TRF for defined features
        rf = functions_analysis.compute_trf(subject=subject, meg_data=meg_data, trial_params=trial_params, trf_params=trf_params, meg_params=meg_params,
                                            features=list(feature_evokeds.keys()), alpha=trf_params['alpha'], use_saved_data=use_saved_data, save_data=save_data,
                                            trf_path=trf_path, trf_fname=trf_fname)

    # Get model coeficients as separate responses to each feature
    feature_evokeds = functions_analysis.parse_trf_to_evoked(subject=subject, rf=rf, meg_data=meg_data, feature_evokeds=feature_evokeds,
                                                             trf_params=trf_params, meg_params=meg_params, sub_idx=sub_idx,
                                                             plot_individuals=plot_individuals, save_fig=save_fig, fig_path=fig_path)

# Grand average
grand_avg = functions_analysis.trf_grand_average(feature_evokeds=feature_evokeds, trf_params=trf_params, meg_params=meg_params,
                                                 display_figs=display_figs, save_fig=save_fig, fig_path=fig_path)