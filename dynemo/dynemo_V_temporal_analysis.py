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

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import paths
import setup
import load
import functions_analysis
from general_utility_functions import cprint, yprint, rprint
from dynemo__utility_functions import find_kept_times_file, nearest_kept_indices, get_subject_trim_start


# ============================================================
# DYNEMO TEMPORAL ANALYSIS
# Event-locked (evoked) response of the DyNeMo mode time courses (alpha).
#
# Events are defined with THIS project's own logic
# (functions_analysis.define_events), exactly like the evoked / TRF analyses,
# so any epoch_id used there works here (fixations, saccades, button presses,
# DA events, etc.). No external "*_meg_samples_to_boxes_sync.csv" is needed.
# ============================================================

################ SETUP ################
use_reweighted_alpha = True

FS_ALPHA = 250            # DyNeMo data sampling rate (Hz)
SMOOTH_WINDOW = 6         # samples (~24 ms) used to smooth the evoked curves
IGNORE_MODE2 = True       # hide the dominant low-frequency mode for raw alpha
RANDOM_STATE = 42

# DyNeMo trimming used in the regression-spectra step (must match dynemo_II)
N_EMBEDDINGS = 15
SEQUENCE_LENGTH = 100

# ------------------------------------------------------------------
# Events to analyse. Each entry's `epoch_id` is passed straight to
# functions_analysis.define_events, so it accepts the same identifiers as the
# evoked / TRF scripts (e.g. 'fix', 'fix_short', 'sac', 'left_but', 'right_but',
# 'DAall', ...). Windows are in seconds relative to the event onset.
# ------------------------------------------------------------------
EVENT_JOBS = {
    "fix":       {"epoch_window": (-1.0, 1.0), "baseline": (-1.0, -0.5), "plot_window": (-1.0, 1.0), "limit": 2500},
    "sac":       {"epoch_window": (-1.0, 1.0), "baseline": (-1.0, -0.5), "plot_window": (-1.0, 1.0), "limit": 2500},
    "left_but":  {"epoch_window": (-5.0, 2.0), "baseline": (1.0, 1.5),   "plot_window": (-2.0, 2.0), "limit": None},
    "right_but": {"epoch_window": (-5.0, 2.0), "baseline": (1.0, 1.5),   "plot_window": (-2.0, 2.0), "limit": None},
}

# Which of the above to actually run this time
events_to_run = ["fix", "left_but", "right_but"]

# ----------------------------
# Paths
# ----------------------------
ALP_PATH = os.path.join(paths.dynemo_infered_parameters_path, "alp.pkl")
DYNEMO_PREPROCESSING = paths.dynemo_preprocessing
TEMPORAL_ANALYSIS = os.path.join(paths.dynemo_temporal_analysis_path, "alpha")
# Figures go to the Plots folder, mirroring the Save data structure
TEMPORAL_PLOTS = os.path.join(paths.dynemo_plots_temporal_analysis_path, "alpha")

if use_reweighted_alpha:
    ALP_PATH = os.path.join(paths.dynemo_infered_parameters_path, "alp_reweighted.pkl")
    TEMPORAL_ANALYSIS = os.path.join(paths.dynemo_temporal_analysis_path, "alpha_reweighted")
    TEMPORAL_PLOTS = os.path.join(paths.dynemo_plots_temporal_analysis_path, "alpha_reweighted")
    IGNORE_MODE2 = False

os.makedirs(TEMPORAL_ANALYSIS, exist_ok=True)
os.makedirs(TEMPORAL_PLOTS, exist_ok=True)

mode_colors = ["tab:blue", "tab:red", "tab:green", "tab:orange",
               "tab:purple", "tab:brown", "tab:pink", "tab:gray"]


################ LOAD ALPHAS ################
cprint(">>> Cargando alphas")
with open(ALP_PATH, "rb") as f:
    alp = pickle.load(f)

exp_info = setup.exp_info()
subjects = exp_info.subjects_ids

cprint(f">>> Sujetos en alpha: {len(alp)}")
cprint(f">>> Sujetos en lista:  {len(subjects)}")

n_modes = alp[0].shape[1]
if IGNORE_MODE2 and n_modes >= 2:
    mode_indices = [m for m in range(n_modes) if m != 1]
else:
    mode_indices = list(range(n_modes))


################ HELPERS ################
def get_event_onsets_seconds(subject, meg_data, epoch_id):
    """Event onset times in seconds (0-based, same time base as kept_times).

    Reuses this project's event-definition logic so buttons, fixations,
    saccades, DA events, etc. are defined exactly as in the evoked / TRF
    analyses. ``define_events`` returns event sample indices that include the
    raw's first_samp, so we subtract ``first_time`` to get 0-based seconds.
    """
    _, events, _, _ = functions_analysis.define_events(subject=subject, meg_data=meg_data, epoch_id=epoch_id)
    if events is None or len(events) == 0:
        return np.array([], dtype=float)
    sfreq = meg_data.info["sfreq"]
    onset_seconds = events[:, 0] / sfreq - meg_data.first_time
    return np.asarray(onset_seconds, dtype=float)


def event_to_alpha_samples(onset_seconds, kept_times, trim_start, n_alpha):
    """Map event onset times (s) to DyNeMo alpha sample indices.

    event onset (s) -> nearest kept MEG sample -> alpha sample (minus trim).
    """
    if len(onset_seconds) == 0:
        return np.array([], dtype=int)

    in_range = (onset_seconds >= kept_times.min()) & (onset_seconds <= kept_times.max())
    onset_seconds = onset_seconds[in_range]
    if len(onset_seconds) == 0:
        return np.array([], dtype=int)

    nearest_idx = nearest_kept_indices(onset_seconds, kept_times)
    alpha_samples = nearest_idx.astype(int) - trim_start

    valid = (alpha_samples >= 0) & (alpha_samples < n_alpha)
    return alpha_samples[valid]


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
    plot_path = os.path.join(TEMPORAL_PLOTS, epoch_id)
    os.makedirs(plot_path, exist_ok=True)

    cprint("\n" + "=" * 80)
    cprint(f">>> Evento: {epoch_id}")
    cprint("=" * 80)

    start_off = int(round(epoch_start * FS_ALPHA))
    end_off = int(round(epoch_end * FS_ALPHA))
    rel_samples = np.arange(start_off, end_off + 1)
    times = rel_samples / FS_ALPHA
    base_mask = (times >= baseline_start) & (times <= baseline_end)

    all_epochs = []   # list of (n_times, n_modes) baseline-corrected windows

    for i, subject_code in enumerate(subjects):

        if i >= len(alp):
            rprint(f">>> No hay alpha para {subject_code}")
            continue

        alpha = alp[i]
        n_alpha = alpha.shape[0]

        # kept_times: 0-based seconds of the 250 Hz raw that survived bad-segment omission
        kept_file = find_kept_times_file(subject_code, DYNEMO_PREPROCESSING)
        if kept_file is None:
            rprint(f">>> No se encontró kept_times para {subject_code}")
            continue
        kept_times = np.load(kept_file)

        # samples trimmed at the start during regression spectra
        trim_start = get_subject_trim_start(subject_code=subject_code,
                                            n_embeddings=N_EMBEDDINGS,
                                            sequence_length=SEQUENCE_LENGTH)

        # Define events with the project's own logic
        subject = setup.subject(subject_id=subject_code)
        meg_data = load.meg(subject_id=subject_code, meg_params={"data_type": "processed"})

        onset_seconds = get_event_onsets_seconds(subject, meg_data, epoch_id)
        centers = event_to_alpha_samples(onset_seconds, kept_times, trim_start, n_alpha)

        # keep only events whose full epoch window fits inside the alpha series
        centers = centers[(centers + start_off >= 0) & (centers + end_off < n_alpha)]

        # optionally subsample (e.g. fixations) to balance subjects
        if limit is not None and len(centers) > limit:
            rng = np.random.default_rng(RANDOM_STATE)
            centers = rng.choice(centers, size=limit, replace=False)

        cprint(f">>> {subject_code}: {len(centers)} eventos válidos ({epoch_id})")

        for c in centers:
            window = alpha[c + start_off: c + end_off + 1, :]   # (n_times, n_modes)
            baseline = window[base_mask].mean(axis=0, keepdims=True)
            all_epochs.append(window - baseline)

    if len(all_epochs) == 0:
        yprint(f">>> No se generaron epochs para {epoch_id}")
        continue

    epochs_arr = np.stack(all_epochs, axis=0)        # (n_epochs, n_times, n_modes)
    evoked = epochs_arr.mean(axis=0)                 # (n_times, n_modes)

    # Smooth each mode's evoked curve
    if SMOOTH_WINDOW and SMOOTH_WINDOW > 1:
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        evoked = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=evoked)

    # Save evoked table
    cols = {"time_from_event": times}
    for m in range(n_modes):
        cols[f"mode_{m + 1}_alpha_change"] = evoked[:, m]
    csv_path = os.path.join(out_path, f"{epoch_id}_evoked_alpha.csv")
    pd.DataFrame(cols).to_csv(csv_path, index=False)
    cprint(f">>> Tabla evoked guardada: {csv_path}")

    ################ PLOT ################
    fig, ax = plt.subplots(figsize=(10, 5))

    for m in mode_indices:
        ax.plot(times, evoked[:, m], color=mode_colors[m % len(mode_colors)],
                label=f"Mode {m + 1}", linewidth=2, alpha=0.95)

    ax.axvline(0, color="gray", linewidth=0.9, alpha=0.6, linestyle="--")
    ax.axvspan(baseline_start, baseline_end, color="gray", alpha=0.15, label="Baseline")
    ax.set_xlim(plot_start, plot_end)
    ax.set_xlabel(f"Time from {epoch_id} (s)")
    ax.set_ylabel("Alpha change from baseline")
    ax.set_title(f"{epoch_id}: DyNeMo alpha evoked response (N={epochs_arr.shape[0]})")
    ax.legend(fontsize=8, ncol=4)
    plt.tight_layout()

    out_fig = os.path.join(plot_path, f"{epoch_id}_evoked_alpha.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    cprint(f">>> Figura guardada: {out_fig}")


cprint(">>> Análisis temporal DyNeMo terminado.")

