import functions_analysis
import functions_general
import load
import setup
import paths
import matplotlib.pyplot as plt
import numpy as np
import plot_general
import save

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
# Same method as MEGEYEDYN's sensor_mtrf.py: one joint TRFlearn model per subject over all channels,
# with ONE ridge regularization shared by all subjects (median of each subject's R²-best).
# Sensor space is DA2 only: DA1 and DA2 used different OPM arrays, so they share no channels.
trf_params = {
    'input_features': {
        'fix': None,# _X_ for intersection between features ['on_mirror', 'stimulus_present', 'on_mirror_X_stimulus_present']
        'sac': None,
        'pur': None,
        'Steering_std_der': None,
        # Speech vs engine-noise contrast: same envelope, masked to the audiobook (Audio) or to DA (engine only).
        # The unmasked 'audio_env_std' is left out because it is the sum of the phase pieces (collinear).
        'audio_env_std_Audio': None,
        'audio_env_std_DA': None,
        'Gas_std_der': None,
        'Brake_std_der': None,
        'left_but': None,
        'right_but': None
    },  # Select features (events)
    # unstandardized impulses (per-event units); kernels fitted on standardized channels
    'regularization': 'median',
    'regularization_grid': np.logspace(-9, 3, 13),
    # Per-feature duration: use dict with 'default' key and optional per-feature overrides
    'tmin': {'default': -0.2, 'Steering_std_der': -2, 'Gas_std_der': -2, 'Brake_std_der': -2, 'left_but': -2, 'right_but': -2},
    'tmax': {'default': 0.5, 'Steering_std_der': 2, 'Gas_std_der': 2, 'Brake_std_der': 2, 'left_but': 2, 'right_but': 2},
    'plot_margin': 0.15,  # seconds to crop from each side of plotted TRF time series
}
# Same fit in both; 'original' rescales each subject's kernels by its channel SDs (T per event)
kernel_units_list = ['standardized', 'original']

time_topos = {
        'fix': 0.7,
        'sac': 1.2,
        'pur': None,
        'Steering_std_der': None,
        'audio_env_std_Audio': 0.1,
        'audio_env_std_DA': 0.1,
        'Gas_std_der': None,
        'Brake_std_der': None,
        'left_but': None,
        'right_but': None
    }

features = functions_analysis.expand_features(trf_params['input_features'])
trf_params['features'] = features
subjects = [s for s in exp_info.subjects_ids if s not in exp_info.DA1_subjects]

# Figure / save paths
_path_tmin = trf_params['tmin'].get('default') if isinstance(trf_params['tmin'], dict) else trf_params['tmin']
_path_tmax = trf_params['tmax'].get('default') if isinstance(trf_params['tmax'], dict) else trf_params['tmax']
fig_path = paths.plots_path + (f"TRF_{meg_params['data_type']}/Band_{meg_params['band_id']}/{functions_general.features_path_str(trf_params['input_features'])}"
                               f"_{_path_tmin}_{_path_tmax}_trflearn_{trf_params['regularization']}/{meg_params['chs_id']}/").replace(":", "")
save_path = fig_path.replace(paths.plots_path, paths.save_path)


def load_subject(subject_id):
    """Band-passed MEG (picked channels) and its bad-annotation-masked feature regressors."""
    subject = setup.subject(subject_id=subject_id)
    meg_data = load.meg(subject_id=subject_id, meg_params=meg_params)
    meg_data = meg_data.pick(functions_general.pick_chs(chs_id=meg_params['chs_id'], info=meg_data.info))
    return meg_data, functions_analysis.build_trf_inputs(subject=subject, meg_data=meg_data, features=features,
                                                         meg_params=meg_params, use_saved_data=use_saved_data)


# Per-subject kernels are cached and reused only when every subject is there
kernels, n_events, target_scales = functions_analysis.cached_trf_group(subjects, load_subject, trf_params, save_path + 'kernels/',
                                                                      overwrite=not use_saved_data)

for units in kernel_units_list:
    units_fig_path = fig_path + f'{units}/'

    feature_evokeds = {feature: [] for feature in features}
    for subject_id in subjects:
        for feature in features:
            kernel = functions_analysis.kernel_units(kernels[subject_id][feature], target_scales[subject_id], units)
            # whole-window baseline, as before
            feature_evokeds[feature].append(kernel.apply_baseline((None, None)))
            if plot_individuals:
                feat_tmin, feat_tmax = functions_analysis.get_feature_tmin_tmax(feature, trf_params)
                margin = trf_params.get('plot_margin', 0)
                fig = kernel.plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(feat_tmin + margin, feat_tmax - margin), titles=feature)
                fig.suptitle(f'{feature}')
                if save_fig:
                    save.fig(fig=fig, fname=f"{feature}_{meg_params['chs_id']}", path=units_fig_path + f'{subject_id}/')

    # Grand average
    grand_avg = functions_analysis.trf_grand_average(feature_evokeds=feature_evokeds, trf_params=trf_params, meg_params=meg_params,
                                                     display_figs=display_figs, save_fig=save_fig, fig_path=units_fig_path)

    # Reduce grand average to z-axis sensors so that the adjacency matrix, the
    # permutation data and the plotted grand average stay aligned.
    for feature in grand_avg.keys():
        grand_avg[feature].pick(functions_general.pick_chs(chs_id='_z', info=grand_avg[feature].info))

    # Run permutations (TRFlearn cluster/TFCE test over channels x lags)
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

    # Plot features figure
    fig = plot_general.plot_trf_features(grand_avg=grand_avg, joint_ylims=None, time_topos=time_topos, top_topos=False,
                                         clusters_mask=clusters_mask,
                                         plot_margin=trf_params.get('plot_margin', 0),
                                         save_fig=save_fig, fig_path=units_fig_path, fname='GA_features_TFCE')
