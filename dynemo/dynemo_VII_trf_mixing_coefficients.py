"""
DyNeMo VI - TRF of the mixing coefficients
==========================================

Temporal Response Function (encoding model) between task features (fixations,
saccades, button presses, continuous signals, ...) and the DyNeMo mixing
coefficients (alpha), treated as if they were ordinary signal channels.

This is the TRF analogue of dynemo_VI_temporal_analysis.py (which computes the
event-locked *evoked* response of alpha). It mirrors the sensor-level mtrf.py,
but runs on the mode time courses instead of MEG sensors, so there is no
spatial topography: each mode is an independent regression target and the
statistics are 1-D cluster permutation tests over the TRF time axis.

Pipeline (per subject):
    alpha (n_time, n_modes) -> continuous 250 Hz mode "raw" (misc channels)
    -> build feature regressors on the same timeline
    -> joint TRFlearn TRF on standardized modes (functions_analysis.fit_trf_group),
       one ridge regularization shared by all subjects (as in MEGEYEDYN)
    -> per-mode TRF EvokedArray, in standardized and original units
Then grand-average across subjects, plot, and run per-mode permutation stats.
"""

import sys
import os

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = r"D:\OneDrive - The University of Nottingham\OPM-MEG-analysis - OPM2\Scripts\dynemo"

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import matplotlib.pyplot as plt
import mne

import paths
import setup
import functions_analysis
from general_utility_functions import cprint, rprint, yprint
import dynemo__mixing_coefficients_utils as mc


# ============================================================
# SETUP
# ============================================================
exp_info = setup.exp_info()

#----- Save data and display figures -----#
save_fig = True
display_figs = True
plot_individuals = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

#----- Mixing coefficients -----#
use_reweighted_alpha = True   # True: normalised/reweighted alpha, False: raw alpha

# DyNeMo trimming used in the regression-spectra step (must match dynemo_II)
from dynemo_config import (n_modes as N_MODES, n_pca as N_PCA, n_embeddings as N_EMBEDDINGS,
                           sequence_length as SEQUENCE_LENGTH, ch_picks as CH_PICKS)

#----- Statistics (TRFlearn cluster/TFCE test; FWE over all modes x lags, as in MEGEYEDYN) -----#
run_permutations = False
pval_threshold = 0.05
# dict -> TFCE; float -> two-tailed p for a fixed cluster-forming t threshold
t_thresh = {"start": 0.0, "step": 0.2}
n_permutations = 1024

#----- Plotting -----#
# Shaded band around each grand-average mode curve: 'sem', 'std', or None
error_band = 'sem'

#----- TRF parameters -----#
# input_features: same identifiers as mtrf.py / define_events.
# Value None = plain feature. (Secondary / phase-tagged variants supported via
# functions_analysis.expand_features, e.g. 'fix': ['on_mirror'].)
# Same method as MEGEYEDYN (functions_analysis.fit_trf_group): one joint TRFlearn
# model per subject, fit on standardized modes, with ONE regularization shared by
# all subjects (median of each subject's R²-best), so kernels are shrunk alike.
trf_params = {
    'input_features': {
        'fix': None,
        'sac': None,
        # 'audio_env_std' dropped: DA1 has no audio, and the shared-regularization fit needs the same features for all
        'left_but': None,
        'right_but': None,
        'Steering_std_der': None,
        'Gas_std_der': None,
        'Brake_std_der': None
    },
    # unstandardized impulses: per-sample variance ~ event rate (1e-4 for rare events).
    # Grid extended below MEGEYEDYN's 1e-6: several DA1 subjects chose the 1e-6 edge.
    'regularization': 'median',
    'regularization_grid': np.logspace(-9, 3, 13),
    # Per-feature windows (seconds). dict with 'default' + per-feature overrides.
    'tmin': {'default': -2, 'fix': -1, 'sac': -1},
    'tmax': {'default': 2, 'fix': 1, 'sac': 1},
    'plot_margin': 0.05,   # seconds cropped from each side when plotting
}
# Same fit in both; 'original' rescales each subject's kernels by its mode SDs
kernel_units_list = ['standardized', 'original']
units_label = {'standardized': 'TRF weight (mode SD per event)',
               'original': 'TRF weight (mixing coefficient per event)'}

#----- Paths -----#
infered_parameters_path = paths.dynemo_run_save_path(N_MODES, N_PCA, N_EMBEDDINGS, SEQUENCE_LENGTH, CH_PICKS, "DyNeMo_Infered_Parameters")
alpha_tag = 'reweighted' if use_reweighted_alpha else 'raw'
features_str = '_'.join(trf_params['input_features'].keys())
run_str = f"alpha_{alpha_tag}/{features_str}/"
# Layout: DyNeMo / modes<..>_pca<..>_emb<..>_seq<..>_<ch_picks> / Mixing_TRF / alpha_<..> / <features>
fig_path = paths.dynemo_run_plots_path(
    N_MODES, N_PCA, N_EMBEDDINGS, SEQUENCE_LENGTH, CH_PICKS, os.path.join("Mixing_TRF", run_str))
save_path = paths.dynemo_run_save_path(
    N_MODES, N_PCA, N_EMBEDDINGS, SEQUENCE_LENGTH, CH_PICKS, os.path.join("DyNeMo_Mixing_TRF", run_str))
os.makedirs(fig_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

# tab10, same as plot_alpha_stack in stage V
mode_colors = [f"C{i}" for i in range(10)]


# ============================================================
# PLOTTING HELPER
# ============================================================
def _plot_mode_trf(evoked, feature, trf_params, colors, title=None,
                   clusters=None, subject_data=None, band=None,
                   save_fig=False, fig_path=None, fname=None, ylabel="TRF weight (a.u.)"):
    """Line plot of the per-mode TRF (one line per mode).

    `clusters[mode]` is the `(start, stop, pvalue)` list from
    mc.mode_cluster_test; significant runs are drawn as coloured bars above
    the traces, one row per mode, labelled with stars and the max cluster p.

    If `subject_data` (n_subjects, n_modes, n_times) and `band` ('sem'/'std')
    are given, a shaded variability band across subjects is drawn around each
    mode's mean curve.
    """
    times = evoked.times
    feat_tmin, feat_tmax = functions_analysis.get_feature_tmin_tmax(feature, trf_params)
    margin = trf_params.get('plot_margin', 0)
    t0, t1 = feat_tmin + margin, feat_tmax - margin
    keep = (times >= t0) & (times <= t1)

    # Variability band across subjects (per mode)
    err = None
    if subject_data is not None and band and subject_data.shape[0] > 1:
        sd = subject_data.std(axis=0, ddof=1)            # (n_modes, n_times)
        if band == 'sem':
            sd = sd / np.sqrt(subject_data.shape[0])
        err = sd

    fig, ax = plt.subplots(figsize=(10, 5))
    for m in range(evoked.data.shape[0]):
        color = colors[m % len(colors)]
        y = evoked.data[m]
        if err is not None:
            ax.fill_between(times[keep], (y - err[m])[keep], (y + err[m])[keep],
                            color=color, alpha=0.15, linewidth=0)
        ax.plot(times[keep], y[keep], color=color,
                linewidth=1.8, alpha=0.95, label=f"Mode {m + 1}")

    ax.axvline(0, color="gray", linewidth=0.9, alpha=0.6, linestyle="--")
    ax.set_xlim(t0, t1)
    if clusters is not None:
        # clip clusters to the plotted window before drawing
        keep_start, keep_stop = np.flatnonzero(keep)[[0, -1]] + [0, 1]
        clipped = [
            [(max(start, keep_start), min(stop, keep_stop), pv)
             for start, stop, pv in found
             if min(stop, keep_stop) > max(start, keep_start)]
            for found in clusters
        ]
        mc.draw_significance_bars(ax, times, clipped, colors)
    ax.set_xlabel(f"Time from {feature} (s)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    plt.tight_layout()

    if save_fig and fig_path and fname:
        mc.save_figure(fig, os.path.join(fig_path, f"{fname}.png"))
    if not display_figs:
        plt.close(fig)
    return fig


# ============================================================
# COMPUTE TRF (one joint model per subject, shared regularization)
# ============================================================
alp = mc.load_alpha(use_reweighted=use_reweighted_alpha,
                    infered_parameters_path=infered_parameters_path)

features = functions_analysis.expand_features(trf_params['input_features'])
trf_params['features'] = features

for subject_id in exp_info.subjects_ids[len(alp):]:
    rprint(f">>> No hay alpha para {subject_id}, lo salto.")
subjects = exp_info.subjects_ids[:len(alp)]


def load_subject(subject_id):
    """Mode "raw" (misc channels, 250 Hz) and its gap-masked feature regressors."""
    subject = setup.subject(subject_id=subject_id)
    mode_raw, valid_mask, mode_times = mc.build_mode_raw(
        subject_code=subject_id, alpha_i=alp[exp_info.subjects_ids.index(subject_id)], ch_picks=CH_PICKS,
        n_pca=N_PCA, n_embeddings=N_EMBEDDINGS, sequence_length=SEQUENCE_LENGTH)
    return mode_raw, {feature: mc.make_mode_trf_input(feature=feature, subject=subject, mode_times=mode_times,
                                                      valid_mask=valid_mask)
                      for feature in features}


kernels, n_events, shared, own, target_scales = functions_analysis.fit_trf_group(subjects, load_subject, trf_params)
np.savez(os.path.join(save_path, "selected_regularization.npz"),
         subjects=np.asarray(subjects),
         own=np.asarray([own[s] for s in subjects], dtype=float),
         shared=shared,
         target_scales=np.stack([target_scales[s] for s in subjects]))
# impulse regressors are 1 at each surviving (unmasked) event
feature_events = {feature: sum(n_events[s][feature] for s in subjects)
                  for feature in features if not mc._is_continuous(feature)}


# ============================================================
# GRAND AVERAGE + STATS + PLOTS (per kernel units)
# ============================================================
n_modes = alp[0].shape[1]

for units in kernel_units_list:
    units_fig_path = os.path.join(fig_path, units)
    units_save_path = os.path.join(save_path, units)
    os.makedirs(units_fig_path, exist_ok=True)
    os.makedirs(units_save_path, exist_ok=True)
    feature_evokeds = {feature: [functions_analysis.kernel_units(kernels[s][feature], target_scales[s], units)
                                 for s in subjects]
                       for feature in features}

    if plot_individuals:
        for feature in features:
            for subject_id, evoked in zip(subjects, feature_evokeds[feature]):
                title = f"{subject_id} - {feature}"
                if feature in feature_events:
                    title += f" ({n_events[subject_id][feature]} events)"
                _plot_mode_trf(evoked, feature, trf_params, mode_colors, title=title,
                               save_fig=save_fig, fig_path=os.path.join(units_fig_path, subject_id),
                               fname=f"{feature}", ylabel=units_label[units])

    for feature in features:
        grand_avg = mne.grand_average(feature_evokeds[feature], interpolate_bads=True)

        # Per-subject data (n_subjects, n_modes, n_times) for variability band + stats
        subject_data = np.array([ev.data for ev in feature_evokeds[feature]])

        # Joint cluster/TFCE test over modes x TRF lags (TRFlearn, FWE across all modes)
        sig_masks = None
        clusters = None
        if run_permutations and len(subject_data) >= 2:
            sig_array, clusters = mc.mode_cluster_test(subject_data, t_thresh, n_permutations=n_permutations,
                                                       pval_threshold=pval_threshold)
            sig_masks = dict(enumerate(sig_array))

        title = f"Grand average - {feature} (N={len(subject_data)} subjects"
        if feature in feature_events:
            title += f", {feature_events[feature]} events"
        _plot_mode_trf(grand_avg, feature, trf_params, mode_colors,
                       title=title + ")",
                       clusters=clusters, subject_data=subject_data, band=error_band,
                       save_fig=save_fig, fig_path=units_fig_path, fname=f"GA_{feature}",
                       ylabel=units_label[units])

        # Persist the TRF results so they can be reloaded without recomputing.
        if save_fig:  # reuse the same "write outputs" switch used for figures
            grand_avg.save(os.path.join(units_save_path, f"GA_{feature}-ave.fif"), overwrite=True)
            sig_array = None
            if sig_masks is not None:
                sig_array = np.array([sig_masks[m] for m in range(n_modes)])  # (n_modes, n_times)
            np.savez(
                os.path.join(units_save_path, f"{feature}_trf.npz"),
                subject_data=subject_data,        # (n_subjects, n_modes, n_times)
                times=grand_avg.times,
                grand_avg=grand_avg.data,         # (n_modes, n_times)
                sig_masks=sig_array if sig_array is not None else np.array([]),
            )
            cprint(f">>> Resultados TRF de '{feature}' ({units}) guardados en {units_save_path}")

cprint(">>> TRF de coeficientes de mezcla terminado.")
