import functions_analysis
import functions_general
import load
import setup
import paths
import matplotlib.pyplot as plt
import os
import plot_general

#----- Path -----#
exp_info = setup.exp_info()

#----- Save data and display figures -----#
use_saved_data = False
save_data = True
save_fig = True
display_figs = True
plot_individuals = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Parameters -----#
trial_params = {}

meg_params = {'chs_id': 'mag_z',
              'band_id': 'Theta',
              'data_type': 'processed',
              'filter_sensors': True,
              }

# TRF parameters
trf_params = {
    'input_features': {
        'fix': None,# _X_ for intersection between features ['on_mirror', 'stimulus_present', 'on_mirror_X_stimulus_present']
        'sac': None,
        'pur': None,
        'Steering_std_der': None,
        'audio_env_std': None,
        'Gas_std_der': None,
        'Brake_std_der': None,
        'left_but': None,
        'right_but': None
    },  # Select features (events)
    'standarize': True,
    'fit_power': False,
    'alpha': [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000],
    # Per-feature duration: use dict with 'default' key and optional per-feature overrides
    # e.g. 'tmin': {'default': -0.2, 'left_but': -2}, 'tmax': {'default': 0.5, 'left_but': 2}
    'tmin': {'default': -0.2, 'Steering_std_der': -2, 'left_but': -2, 'right_but': -2},
    'tmax': {'default': 0.5, 'Steering_std_der': 2, 'left_but': 2, 'right_but': 2},
    'plot_margin': 0.15,  # seconds to crop from each side of plotted TRF time series
}

time_topos = {
        'fix': 0.7,# _X_ for intersection between features ['on_mirror', 'stimulus_present', 'on_mirror_X_stimulus_present']
        'sac': 1.2,
        'pur': None,
        'Steering_std_der': None,
        'audio_env_std': 0.0,
        'Gas_std_der': None,
        'Brake_std_der': None,
        'left_but': None,
        'right_but': None
    }


# Figure path
_path_tmin = trf_params['tmin'].get('default') if isinstance(trf_params['tmin'], dict) else trf_params['tmin']
_path_tmax = trf_params['tmax'].get('default') if isinstance(trf_params['tmax'], dict) else trf_params['tmax']
_path_bline = (_path_tmin, _path_tmax)
_alpha_tag = f'_alpha{trf_params["alpha"]}' if trf_params['alpha'] is None else '_alphaCV'
fig_path = paths.plots_path + (f"TRF_{meg_params['data_type']}/Band_{meg_params['band_id']}/{functions_general.features_path_str(trf_params['input_features'])}"
                               f"_{_path_tmin}_{_path_tmax}_bline{_path_bline}{_alpha_tag}_"
                               f"std{trf_params['standarize']}/{meg_params['chs_id']}/").replace(":", "")

# Change path to include envelope power
if trf_params['fit_power']:
    fig_path = fig_path.replace(f"TRF_{meg_params['data_type']}", f"TRF_{meg_params['data_type']}_ENV")

# Save path
save_path = fig_path.replace(paths.plots_path, paths.save_path)

# Define Grand average variables
feature_evokeds = {}

features = functions_analysis.expand_features(trf_params['input_features'])
for feature in features:
    feature_evokeds[feature] = []

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
        rf = functions_analysis.compute_trf(subject=subject, meg_data=meg_data, trf_params=trf_params, meg_params=meg_params,
                                            features=list(feature_evokeds.keys()), alpha=trf_params['alpha'], use_saved_data=use_saved_data, save_data=save_data,
                                            trf_path=trf_path, trf_fname=trf_fname)

    # Get model coeficients as separate responses to each feature
    feature_evokeds = functions_analysis.parse_trf_to_evoked(subject=subject, rf=rf, meg_data=meg_data, feature_evokeds=feature_evokeds,
                                                             trf_params=trf_params, meg_params=meg_params, sub_idx=sub_idx,
                                                             plot_individuals=plot_individuals, save_fig=save_fig, fig_path=fig_path)

# Grand average
grand_avg = functions_analysis.trf_grand_average(feature_evokeds=feature_evokeds, trf_params=trf_params, meg_params=meg_params,
                                                 display_figs=display_figs, save_fig=save_fig, fig_path=fig_path)

joint_ylims = None
# Plot features figure
fname = f'GA_features_TFCE'

for feature in grand_avg.keys():
    grand_avg[feature].pick(functions_general.pick_chs(chs_id='_z', info=grand_avg[feature].info))


fig = plot_general.plot_trf_features(grand_avg=grand_avg, joint_ylims=joint_ylims, time_topos=time_topos, top_topos=False,
                                     plot_margin=trf_params.get('plot_margin', 0),
                                     save_fig=save_fig, fig_path=fig_path, fname=fname)
