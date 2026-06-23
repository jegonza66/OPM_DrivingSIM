import functions_analysis
import functions_general
import load
import setup
import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import plot_general

#----- Path -----#
exp_info = setup.exp_info()

#----- Save data and display figures -----#
use_saved_data = True
save_data = True
save_fig = True
display_figs = True
plot_individuals = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

#----- Statistics (cluster-based permutations) -----#
run_permutations = True
pval_threshold = 0.05          # significance level for clusters
t_thresh = dict(start=0, step=0.2)  # TFCE; or a float for a fixed t-threshold
n_permutations = 1024

#-----  Parameters -----#
trial_params = {}

meg_params = {'chs_id': 'mag_z',
              'band_id': (0.1, 40),
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
    # Alpha cross-validation: k-fold over contiguous temporal blocks.
    # cv_aggregate: 'mean_fisher' (default, Fisher-z averaged per-fold correlation),
    #               'mean' (plain average), or 'pool' (one correlation over pooled preds)
    'cv_n_splits': 5,
    'cv_aggregate': 'mean_fisher',
    # Per-feature duration: use dict with 'default' key and optional per-feature overrides
    # e.g. 'tmin': {'default': -0.2, 'left_but': -2}, 'tmax': {'default': 0.5, 'left_but': 2}
    'tmin': {'default': -0.2, 'Steering_std_der': -2, 'Gas_std_der': -2, 'Brake_std_der': -2, 'left_but': -2, 'right_but': -2},
    'tmax': {'default': 0.5, 'Steering_std_der': 2, 'Gas_std_der': 2, 'Brake_std_der': 2, 'left_but': 2, 'right_but': 2},
    'plot_margin': 0.15,  # seconds to crop from each side of plotted TRF time series
}

time_topos = {
        'fix': 0.7,
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


# Reduce grand average to z-axis sensors so that the adjacency matrix, the
# permutation data and the plotted grand average stay aligned.
for feature in grand_avg.keys():
    grand_avg[feature].pick(functions_general.pick_chs(chs_id='_z', info=grand_avg[feature].info))

# Run permutations
if run_permutations:
    clusters_mask = {}
    clusters_pvalues = {}

    # Channels present in EVERY subject's evoked (intersection across all subjects and features)
    channel_sets = [set(ev.ch_names) for ev_list in feature_evokeds.values() for ev in ev_list]
    common = set.intersection(*channel_sets)
    stat_chs = [ch for ch in grand_avg[list(grand_avg.keys())[0]].ch_names if ch in common]

    ch_adjacency_sparse = functions_general.get_channel_adjacency(info=grand_avg[list(grand_avg.keys())[0]].info, ch_type='mag', picks=stat_chs)

    for feature in grand_avg.keys():
        print('Running permutations test for feature:', feature)
        grand_avg[feature].pick(stat_chs)
        data = np.array([ev.copy().pick(stat_chs).data.T for ev in feature_evokeds[feature]])
        clusters_mask_transp, clusters_pvalues[feature] = functions_analysis.run_permutations_test(data=data, pval_threshold=pval_threshold, t_thresh=t_thresh, adj_matrix=ch_adjacency_sparse, n_permutations=n_permutations)
        clusters_mask[feature] = clusters_mask_transp.T
else:
    clusters_mask = None

joint_ylims = None
# Plot features figure
fname = f'GA_features_TFCE'

fig = plot_general.plot_trf_features(grand_avg=grand_avg, joint_ylims=joint_ylims, time_topos=time_topos, top_topos=False,
                                     clusters_mask=clusters_mask,
                                     plot_margin=trf_params.get('plot_margin', 0),
                                     save_fig=save_fig, fig_path=fig_path, fname=fname)
