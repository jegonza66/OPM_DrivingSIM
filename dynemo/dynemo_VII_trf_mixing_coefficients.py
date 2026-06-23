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
    -> ridge TRF (mne ReceptiveField via functions_analysis.fit_mtrf, chs_id='misc')
    -> per-mode TRF EvokedArray
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
N_MODES = 6
N_EMBEDDINGS = 15
SEQUENCE_LENGTH = 100

#----- Statistics (1-D temporal cluster permutations, per mode) -----#
run_permutations = True
pval_threshold = 0.05
t_thresh = dict(start=0, step=0.2)   # TFCE; or a float for a fixed t-threshold
n_permutations = 1024

#----- Plotting -----#
# Shaded band around each grand-average mode curve: 'sem', 'std', or None
error_band = 'sem'

#----- TRF parameters -----#
# input_features: same identifiers as mtrf.py / define_events.
# Value None = plain feature. (Secondary / phase-tagged variants supported via
# functions_analysis.expand_features, e.g. 'fix': ['on_mirror'].)
trf_params = {
    'input_features': {
        'fix': None,
        'sac': None,
        'audio_env_std': None,
        'left_but': None,
        'right_but': None,
        'Steering_std_der': None,
        'Gas_std_der': None,
        'Brake_std_der': None
    },
    'standarize': True,
    'alpha': [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000],   # ridge; list -> CV
    # Per-feature windows (seconds). dict with 'default' + per-feature overrides.
    'tmin': {'default': -2, 'fix': -1, 'sac': -1, 'audio_env_std': -0.2},
    'tmax': {'default': 2, 'fix': 1, 'sac': 1, 'audio_env_std': 0.5},
    'plot_margin': 0.05,   # seconds cropped from each side when plotting
    'fit_power': False,    # mode time courses are amplitudes already
}

#----- Paths -----#
infered_parameters_path = paths.dynemo_run_save_path(N_MODES, N_EMBEDDINGS, SEQUENCE_LENGTH, "DyNeMo_Infered_Parameters")
alpha_tag = 'reweighted' if use_reweighted_alpha else 'raw'
features_str = '_'.join(trf_params['input_features'].keys())
run_str = f"alpha_{alpha_tag}/{features_str}/"
# Layout: DyNeMo / emb<..>_seq<..> / Mixing_TRF / alpha_<..> / <features>
fig_path = paths.dynemo_run_plots_path(
    N_MODES, N_EMBEDDINGS, SEQUENCE_LENGTH, os.path.join("Mixing_TRF", run_str))
save_path = paths.dynemo_run_save_path(
    N_MODES, N_EMBEDDINGS, SEQUENCE_LENGTH, os.path.join("DyNeMo_Mixing_TRF", run_str))
os.makedirs(fig_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

mode_colors = ["tab:blue", "tab:red", "tab:green", "tab:orange",
               "tab:purple", "tab:brown", "tab:pink", "tab:gray",
               "tab:olive", "tab:cyan"]


# ============================================================
# PLOTTING HELPER
# ============================================================
def _plot_mode_trf(evoked, feature, trf_params, colors, title=None,
                   sig_masks=None, subject_data=None, band=None,
                   save_fig=False, fig_path=None, fname=None):
    """Line plot of the per-mode TRF (one line per mode).

    Significant time samples (per-mode 1-D cluster permutation) are marked as a
    thick overlay on each mode's curve when `sig_masks` is provided.

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
        if sig_masks is not None and m in sig_masks:
            sig = sig_masks[m] & keep
            if np.any(sig):
                # Trace the curve in black ONLY on significant samples; NaN
                # elsewhere so matplotlib breaks the line instead of
                # interpolating across non-significant gaps.
                y_sig = np.where(sig, y, np.nan)
                ax.plot(times, y_sig, color="black", linewidth=2.5,
                        alpha=0.9, solid_capstyle="round")

    ax.axvline(0, color="gray", linewidth=0.9, alpha=0.6, linestyle="--")
    ax.set_xlim(t0, t1)
    ax.set_xlabel(f"Time from {feature} (s)")
    ax.set_ylabel("TRF weight (a.u.)")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, ncol=4)
    plt.tight_layout()

    if save_fig and fig_path and fname:
        os.makedirs(fig_path, exist_ok=True)
        fig.savefig(os.path.join(fig_path, f"{fname}.png"), dpi=300)
    if not display_figs:
        plt.close(fig)
    return fig


# ============================================================
# COMPUTE TRF PER SUBJECT
# ============================================================
alp = mc.load_alpha(use_reweighted=use_reweighted_alpha,
                    infered_parameters_path=infered_parameters_path)

features = functions_analysis.expand_features(trf_params['input_features'])
feature_evokeds = {feature: [] for feature in features}

for sub_idx, subject_id in enumerate(exp_info.subjects_ids):

    if sub_idx >= len(alp):
        rprint(f">>> No hay alpha para {subject_id}, lo salto.")
        continue

    cprint("\n" + "=" * 80)
    cprint(f">>> Sujeto: {subject_id}")
    cprint("=" * 80)

    subject = setup.subject(subject_id=subject_id)

    # Build the continuous mode "raw" (misc channels) on the 250 Hz timeline
    mode_raw, valid_mask, mode_times = mc.build_mode_raw(
        subject_code=subject_id, alpha_i=alp[sub_idx],
        n_embeddings=N_EMBEDDINGS, sequence_length=SEQUENCE_LENGTH)

    # Build feature regressors on the same timeline
    input_arrays = {}
    for feature in features:
        input_arrays[feature] = mc.make_mode_trf_input(
            feature=feature, subject=subject, mode_times=mode_times,
            valid_mask=valid_mask)

    # Group features by (tmin, tmax) so each duration is fit with its own model
    duration_groups = functions_analysis.group_features_by_duration(features, trf_params)

    rf_results = []
    for (group_tmin, group_tmax), group_features in duration_groups.items():
        group_input = np.array([input_arrays[f] for f in group_features]).T  # (n_times, n_feat)

        cprint(f">>> Ajustando TRF (modos) tmin={group_tmin}, tmax={group_tmax} "
               f"para {group_features}")
        group_rf = functions_analysis.fit_mtrf(
            meg_data=mode_raw, tmin=group_tmin, tmax=group_tmax,
            alpha=trf_params['alpha'], fit_power=trf_params['fit_power'],
            model_input=group_input, chs_id='misc',
            standarize=trf_params['standarize'], n_jobs=4)

        rf_results.append({'rf': group_rf, 'features': group_features,
                           'tmin': group_tmin, 'tmax': group_tmax,
                           'best_alpha': functions_analysis.extract_best_alpha(group_rf)})

    # Map each feature to its group RF + column index
    feature_map = {}
    for group in rf_results:
        for col, feat in enumerate(group['features']):
            feature_map[feat] = {'rf': group['rf'], 'col': col,
                                 'tmin': group['tmin'], 'tmax': group['tmax']}

    # Build a per-mode TRF EvokedArray for each feature
    for feature in features:
        fmap = feature_map[feature]
        # ReceptiveField.coef_ shape: (n_modes, n_features, n_delays)
        trf = fmap['rf'].coef_[:, fmap['col'], :]   # (n_modes, n_delays)
        evoked = mne.EvokedArray(data=trf, info=mode_raw.info,
                                 tmin=fmap['tmin'], baseline=(fmap['tmin'], fmap['tmax']))
        feature_evokeds[feature].append(evoked)

        if plot_individuals:
            _plot_mode_trf(evoked, feature, trf_params, mode_colors,
                           title=f"{subject_id} - {feature}",
                           save_fig=save_fig,
                           fig_path=os.path.join(fig_path, subject_id),
                           fname=f"{feature}")


# ============================================================
# GRAND AVERAGE + STATS + PLOTS
# ============================================================
n_modes = alp[0].shape[1]

for feature in features:
    if len(feature_evokeds[feature]) == 0:
        yprint(f">>> Sin datos para {feature}, lo salto.")
        continue

    grand_avg = mne.grand_average(feature_evokeds[feature], interpolate_bads=True)

    # Per-subject data (n_subjects, n_modes, n_times) for variability band + stats
    subject_data = np.array([ev.data for ev in feature_evokeds[feature]])

    # Per-mode permutation stats over the TRF time axis
    sig_masks = None
    if run_permutations:
        sig_masks = {}
        for m in range(n_modes):
            # (n_subjects, n_times) for this mode
            data = subject_data[:, m, :]
            sig_masks[m] = mc.temporal_cluster_test(
                data=data, t_thresh=t_thresh, n_permutations=n_permutations,
                pval_threshold=pval_threshold)

    _plot_mode_trf(grand_avg, feature, trf_params, mode_colors,
                   title=f"Grand average - {feature} (N={len(feature_evokeds[feature])})",
                   sig_masks=sig_masks, subject_data=subject_data, band=error_band,
                   save_fig=save_fig, fig_path=fig_path, fname=f"GA_{feature}")

    # Persist the TRF results so they can be reloaded without recomputing.
    if save_fig:  # reuse the same "write outputs" switch used for figures
        # Grand-average TRF as an MNE Evoked (-ave.fif)
        grand_avg.save(os.path.join(save_path, f"GA_{feature}-ave.fif"),
                       overwrite=True)
        # Per-subject curves, time axis and significance mask as arrays
        sig_array = None
        if sig_masks is not None:
            sig_array = np.array([sig_masks[m] for m in range(n_modes)])  # (n_modes, n_times)
        np.savez(
            os.path.join(save_path, f"{feature}_trf.npz"),
            subject_data=subject_data,        # (n_subjects, n_modes, n_times)
            times=grand_avg.times,
            grand_avg=grand_avg.data,         # (n_modes, n_times)
            sig_masks=sig_array if sig_array is not None else np.array([]),
        )
        cprint(f">>> Resultados TRF de '{feature}' guardados en {save_path}")

cprint(">>> TRF de coeficientes de mezcla terminado.")


