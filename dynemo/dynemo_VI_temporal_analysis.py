import sys
import os

# Folder containing this file's modules. Use __file__ when run as a script,
# otherwise fall back to the known path (e.g. when run in the Python console).
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = r"D:\OneDrive - The University of Nottingham\OPM-MEG-analysis - OPM2\Scripts\dynemo"

sys.path.insert(0, HERE)                 # dynemo__utility_functions
sys.path.insert(0, os.path.dirname(HERE))  # paths, load, setup, ...

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

import paths
import setup
import load
from general_utility_functions import cprint, yprint, rprint
import dynemo__mixing_coefficients_utils as mc


# ============================================================
# DYNEMO TEMPORAL ANALYSIS  (MNE version)
# ------------------------------------------------------------
# Event-locked (evoked) response of the DyNeMo mode time courses (alpha),
# implemented with native MNE objects. This is a 1:1 translation of
# dynemo_VI_temporal_analysis.py meant for side-by-side comparison.
#
# Difference vs. the original V module:
#   - The original epochs the alpha *in alpha-index space* (windows can straddle
#     bad-segment gaps, treating concatenated samples as contiguous).
#   - Here each subject's alpha is mapped back onto a continuous 250 Hz
#     `RawArray` (one `misc` channel per mode) via mc.build_mode_raw, the
#     trimmed/bad-segment gaps are flagged from `valid_mask` as 'BAD_gap'
#     annotations, and epochs are cut with mne.Epochs. With
#     `reject_gap_epochs=True` (default) any epoch overlapping a gap is dropped,
#     which is the cleaner behaviour the valid mask enables. Set it to False to
#     keep all epochs (gap samples then contribute zeros) for a closer count
#     match to V.
# Everything else (events, windows, baseline, smoothing, stats, plots) is
# identical to the original module.
# ============================================================

################ SETUP ################
use_reweighted_alpha = True

FS_ALPHA = 250            # DyNeMo data sampling rate (Hz)
SMOOTH_WINDOW = 6         # samples (~24 ms) used to smooth the evoked curves
IGNORE_MODE2 = True       # hide the dominant low-frequency mode for raw alpha
RANDOM_STATE = 42
reject_gap_epochs = True  # drop epochs overlapping trimmed / bad-segment gaps

# Statistics: 1-D temporal cluster permutation across subjects (per mode)
run_permutations = True
pval_threshold = 0.05
t_thresh = dict(start=0, step=0.2)   # TFCE; or a float for a fixed t-threshold
n_permutations = 1024

# Shaded band around each grand-average mode curve: 'sem', 'std', or None
error_band = 'sem'

# DyNeMo trimming used in the regression-spectra step (must match dynemo_II)
N_EMBEDDINGS = 15
SEQUENCE_LENGTH = 100

# ------------------------------------------------------------------
# Events to analyse (same identifiers as the original V module).
# ------------------------------------------------------------------
EVENT_JOBS = {
    "fix":       {"epoch_window": (-1.0, 1.0), "baseline": (-1.0, -0.5), "plot_window": (-1.0, 1.0), "limit": 2500},
    "sac":       {"epoch_window": (-1.0, 1.0), "baseline": (-1.0, -0.5), "plot_window": (-1.0, 1.0), "limit": 2500},
    "left_but":  {"epoch_window": (-2.0, 2.0), "baseline": (1.0, 1.5),   "plot_window": (-2.0, 2.0), "limit": None},
    "right_but": {"epoch_window": (-2.0, 2.0), "baseline": (1.0, 1.5),   "plot_window": (-2.0, 2.0), "limit": None}
}

# Which of the above to actually run this time
events_to_run = list(EVENT_JOBS.keys())

# ----------------------------
# Paths
# ----------------------------
alpha_subdir = "alpha_reweighted" if use_reweighted_alpha else "alpha"
TEMPORAL_ANALYSIS = os.path.join(paths.dynemo_temporal_analysis_path, alpha_subdir)
TEMPORAL_PLOTS = os.path.join(paths.dynemo_plots_temporal_analysis_path, alpha_subdir)
if use_reweighted_alpha:
    IGNORE_MODE2 = False

os.makedirs(TEMPORAL_ANALYSIS, exist_ok=True)
os.makedirs(TEMPORAL_PLOTS, exist_ok=True)

mode_colors = ["tab:blue", "tab:red", "tab:green", "tab:orange",
               "tab:purple", "tab:brown", "tab:pink", "tab:gray"]


################ LOAD ALPHAS ################
alp = mc.load_alpha(use_reweighted=use_reweighted_alpha)

exp_info = setup.exp_info()
subjects = exp_info.subjects_ids

cprint(f">>> Sujetos en alpha: {len(alp)}")
cprint(f">>> Sujetos en lista:  {len(subjects)}")

n_modes = alp[0].shape[1]
if IGNORE_MODE2 and n_modes >= 2:
    mode_indices = [m for m in range(n_modes) if m != 1]
else:
    mode_indices = list(range(n_modes))


################ EVENT-LOCKED ALPHA ################
for epoch_id in events_to_run:

    if epoch_id not in EVENT_JOBS:
        yprint(f">>> Evento '{epoch_id}' sin configuración en EVENT_JOBS, lo salto.")
        continue

    job = EVENT_JOBS[epoch_id]
    epoch_start, epoch_end = job["epoch_window"]
    baseline_start, baseline_end = job["baseline"]
    plot_start, plot_end = job["plot_window"]
    limit = job.get("limit")

    out_path = os.path.join(TEMPORAL_ANALYSIS, epoch_id)
    os.makedirs(out_path, exist_ok=True)

    cprint("\n" + "=" * 80)
    cprint(f">>> Evento: {epoch_id}")
    cprint("=" * 80)

    all_epochs = []       # list of (n_epochs, n_modes, n_times) per subject
    subject_evokeds = []  # per-subject mean window (n_times, n_modes) for group stats
    times = None          # epoch time vector (set from the first valid subject)

    for i, subject_code in enumerate(subjects):

        if i >= len(alp):
            rprint(f">>> No hay alpha para {subject_code}")
            continue

        # Continuous 250 Hz mode "raw" (misc channels) + validity mask
        mode_raw, valid_mask, _ = mc.build_mode_raw(
            subject_code=subject_code, alpha_i=alp[i],
            n_embeddings=N_EMBEDDINGS, sequence_length=SEQUENCE_LENGTH)
        fs = mode_raw.info["sfreq"]
        n_times_full = len(mode_raw.times)

        # Flag trimmed / bad-segment gaps so overlapping epochs can be rejected
        if reject_gap_epochs:
            mode_raw.set_annotations(mc.bad_annotations_from_mask(valid_mask, fs))

        # Define events with the project's own logic (0-based seconds)
        subject = setup.subject(subject_id=subject_code)
        meg_data = load.meg(subject_id=subject_code, meg_params={"data_type": "processed"})
        onset_seconds = mc.get_event_onsets_seconds(subject, meg_data, epoch_id)

        if len(onset_seconds) == 0:
            cprint(f">>> {subject_code}: 0 eventos ({epoch_id})")
            continue

        # Onsets (s) -> mode-raw sample indices
        ev_samples = np.round(onset_seconds * fs).astype(int)
        ev_samples = ev_samples[(ev_samples >= 0) & (ev_samples < n_times_full)]
        ev_samples = np.unique(ev_samples)

        # optionally subsample (e.g. fixations) to balance subjects
        if limit is not None and len(ev_samples) > limit:
            rng = np.random.default_rng(RANDOM_STATE)
            ev_samples = np.sort(rng.choice(ev_samples, size=limit, replace=False))

        events = np.column_stack([ev_samples,
                                  np.zeros_like(ev_samples),
                                  np.ones_like(ev_samples)]).astype(int)

        # NOTE: mne.Epochs(baseline=...) does NOT rescale 'misc' channels, so we
        # epoch WITHOUT baseline here and subtract the baseline-window mean
        # manually below (exactly like the original numpy module).
        epochs = mne.Epochs(mode_raw, events=events, event_id={epoch_id: 1},
                            tmin=epoch_start, tmax=epoch_end,
                            baseline=None,
                            picks="misc", preload=True, proj=False,
                            event_repeated="drop",
                            reject_by_annotation=reject_gap_epochs, verbose=False)

        if len(epochs) == 0:
            cprint(f">>> {subject_code}: 0 epochs válidos ({epoch_id})")
            continue

        cprint(f">>> {subject_code}: {len(epochs)} epochs válidos ({epoch_id})")

        if times is None:
            times = epochs.times.copy()

        data = epochs.get_data()                  # (n_epochs, n_modes, n_times)
        # Manual baseline correction (misc channels are skipped by MNE baseline)
        base_mask = (epochs.times >= baseline_start) & (epochs.times <= baseline_end)
        data = data - data[:, :, base_mask].mean(axis=2, keepdims=True)
        all_epochs.append(data)
        subject_evokeds.append(data.mean(axis=0).T)   # (n_times, n_modes)

    if len(all_epochs) == 0:
        yprint(f">>> No se generaron epochs para {epoch_id}")
        continue

    epochs_arr = np.concatenate(all_epochs, axis=0)   # (n_epochs, n_modes, n_times)
    evoked = epochs_arr.mean(axis=0).T                # (n_times, n_modes)

    # Smooth each mode's evoked curve
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW if (SMOOTH_WINDOW and SMOOTH_WINDOW > 1) else None
    if kernel is not None:
        evoked = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=evoked)

    ################ STATISTICS (per-mode temporal cluster permutation) ################
    # Per-subject smoothed curves (shared by the variability band and the stats)
    subj_arr = None
    if len(subject_evokeds) >= 2:
        subj_arr = np.stack(subject_evokeds, axis=0)   # (n_subjects, n_times, n_modes)
        # Smooth each subject's curve like the grand mean so band/stats match the plot
        if kernel is not None:
            subj_arr = np.apply_along_axis(
                lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=subj_arr)

    # Variability band across subjects (per mode): (n_times, n_modes)
    band_err = None
    if subj_arr is not None and error_band:
        band_err = subj_arr.std(axis=0, ddof=1)
        if error_band == 'sem':
            band_err = band_err / np.sqrt(subj_arr.shape[0])

    sig_masks = None
    if run_permutations and subj_arr is not None:
        sig_masks = {}
        for m in range(n_modes):
            sig_masks[m] = mc.temporal_cluster_test(
                data=subj_arr[:, :, m], t_thresh=t_thresh,
                n_permutations=n_permutations, pval_threshold=pval_threshold)
        cprint(f">>> Permutaciones temporales hechas con N={subj_arr.shape[0]} sujetos")
    elif run_permutations:
        yprint(">>> Muy pocos sujetos con eventos para estadística; la salto.")

    # Save evoked table
    cols = {"time_from_event": times}
    for m in range(n_modes):
        cols[f"mode_{m + 1}_alpha_change"] = evoked[:, m]
        if sig_masks is not None:
            cols[f"mode_{m + 1}_significant"] = sig_masks[m]
    csv_path = os.path.join(out_path, f"{epoch_id}_evoked_alpha.csv")
    pd.DataFrame(cols).to_csv(csv_path, index=False)
    cprint(f">>> Tabla evoked guardada: {csv_path}")

    ################ PLOT ################
    fig, ax = plt.subplots(figsize=(10, 5))

    for m in mode_indices:
        color = mode_colors[m % len(mode_colors)]
        # Variability band across subjects
        if band_err is not None:
            ax.fill_between(times, evoked[:, m] - band_err[:, m], evoked[:, m] + band_err[:, m],
                            color=color, alpha=0.15, linewidth=0)
        ax.plot(times, evoked[:, m], color=color,
                label=f"Mode {m + 1}", linewidth=2, alpha=0.95)
        # Highlight significant time samples for this mode: trace the curve in
        # black only where significant (NaN elsewhere so the line breaks at gaps
        # instead of interpolating across them).
        if sig_masks is not None and np.any(sig_masks[m]):
            y_sig = np.where(sig_masks[m], evoked[:, m], np.nan)
            ax.plot(times, y_sig, color="black", linewidth=2.5,
                    alpha=0.9, solid_capstyle="round")

    ax.axvline(0, color="gray", linewidth=0.9, alpha=0.6, linestyle="--")
    ax.axvspan(baseline_start, baseline_end, color="gray", alpha=0.15, label="Baseline")
    ax.set_xlim(plot_start, plot_end)
    ax.set_xlabel(f"Time from {epoch_id} (s)")
    ax.set_ylabel("Alpha change from baseline")
    ax.set_title(f"{epoch_id}: DyNeMo alpha evoked response (MNE, N={epochs_arr.shape[0]})")
    ax.legend(fontsize=8, ncol=4)
    plt.tight_layout()

    out_fig = os.path.join(TEMPORAL_PLOTS, f"{epoch_id}_evoked_alpha.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    cprint(f">>> Figura guardada: {out_fig}")


cprint(">>> Análisis temporal DyNeMo (MNE) terminado.")


