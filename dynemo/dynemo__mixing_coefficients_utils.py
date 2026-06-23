"""
Utilities to treat the DyNeMo mixing coefficients (alpha) as if they were
ordinary signal channels, aligned to the recording timeline, so that the
project's evoked / TRF machinery can be reused on them.

The mixing coefficients live on the DyNeMo timeline:
    raw (processed) -> filter + resample to 250 Hz -> drop BAD segments
    -> (kept_times) -> DyNeMo trims `trim_start` samples at the start -> alpha

To run a TRF (or any event-locked analysis) we need the alpha time course on a
*continuous* 250 Hz timeline that shares the recording's 0-based seconds, plus
a validity mask flagging which samples actually carry an alpha value (i.e. not a
trimmed / bad-segment gap). Everything here is kept in plain 0-based seconds to
avoid first_samp / annotation offset pitfalls (same convention as
dynemo_VI_temporal_analysis.py).
"""

import sys
import os

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = r"D:\OneDrive - The University of Nottingham\OPM-MEG-analysis - OPM2\Scripts\dynemo"

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import pickle
import numpy as np
import mne

import paths
import load
import functions_analysis
from general_utility_functions import cprint, yprint
from dynemo__utility_functions import (find_kept_times_file,
                                       get_subject_trim_start)

FS_ALPHA = 250  # DyNeMo sampling rate (Hz)

# Feature names that are CONTINUOUS regressors (read from the processed raw and
# resampled to the alpha timeline). Anything else is treated as an event /
# impulse feature defined via functions_analysis.define_events.
_CONTINUOUS_KEYS = ("Steering", "Gas", "Brake", "audio_env")


# ------------------------------------------------------------------
# Alpha loading
# ------------------------------------------------------------------
def load_alpha(use_reweighted=True, infered_parameters_path=None):
    """Load the per-subject mixing coefficients.

    Returns a list of (n_time, n_modes) arrays.

    `infered_parameters_path` selects the (per-run) folder that holds the
    alp*.pkl files. When None it falls back to the legacy shared location.
    """
    if infered_parameters_path is None:
        infered_parameters_path = paths.dynemo_infered_parameters_path
    fname = "alp_reweighted.pkl" if use_reweighted else "alp.pkl"
    alp_path = os.path.join(infered_parameters_path, fname)
    cprint(f">>> Cargando alphas desde {alp_path}")
    with open(alp_path, "rb") as f:
        alp = pickle.load(f)
    return alp


# ------------------------------------------------------------------
# Mode time course -> continuous 250 Hz timeline
# ------------------------------------------------------------------
def build_mode_raw(subject_code, alpha_i, n_embeddings=15, sequence_length=100,
                   mode_prefix="mode"):
    """Map a subject's alpha (n_time, n_modes) onto the full 250 Hz timeline.

    Returns
    -------
    mode_raw : mne.io.RawArray
        Continuous raw at 250 Hz with one `misc` channel per mode
        (`mode_1 ... mode_n`). Trimmed / bad-segment samples are 0.
    valid_mask : ndarray, shape (n_times,)
        1 where an alpha value was placed, 0 in gaps (trimmed / bad segments).
    times : ndarray, shape (n_times,)
        0-based seconds of the continuous timeline.
    """
    # The DyNeMo-preprocessed raw (1-45 Hz, 250 Hz) defines the full timeline,
    # its length and its BAD annotations.
    preproc_fif = os.path.join(paths.dynemo_preprocessing, subject_code,
                               "preprocessed",
                               f"{subject_code}_preproc_1-45Hz_250Hz-raw.fif")
    if not os.path.exists(preproc_fif):
        raise FileNotFoundError(
            f"No encuentro el raw preprocesado de DyNeMo para {subject_code}:\n"
            f"  {preproc_fif}")
    preproc_raw = mne.io.read_raw_fif(preproc_fif, verbose=False)
    fs = preproc_raw.info["sfreq"]
    n_times_full = len(preproc_raw.times)

    # kept_times: 0-based seconds of the 250 Hz samples that survived BAD omission
    kept_file = find_kept_times_file(subject_code, paths.dynemo_preprocessing)
    if kept_file is None:
        raise FileNotFoundError(f"No encuentro kept_times para {subject_code}")
    kept_times = np.load(kept_file)

    # Samples DyNeMo trimmed at the start (regression-spectra trimming)
    trim_start = get_subject_trim_start(subject_code=subject_code,
                                        n_modes=alpha_i.shape[1],
                                        n_embeddings=n_embeddings,
                                        sequence_length=sequence_length)

    n_alpha, n_modes = alpha_i.shape

    # alpha sample j  <->  kept sample (j + trim_start)  <->  kept_times[...] seconds
    kept_idx = np.arange(trim_start, trim_start + n_alpha)
    valid = kept_idx < len(kept_times)
    times_sec = kept_times[kept_idx[valid]]
    full_idx = preproc_raw.time_as_index(times_sec, use_rounding=True)
    inb = full_idx < n_times_full
    full_idx = full_idx[inb]

    data = np.zeros((n_modes, n_times_full), dtype=np.float64)
    data[:, full_idx] = alpha_i[valid][inb].T

    valid_mask = np.zeros(n_times_full, dtype=np.float64)
    valid_mask[full_idx] = 1.0

    ch_names = [f"{mode_prefix}_{m + 1}" for m in range(n_modes)]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="misc")
    mode_raw = mne.io.RawArray(data, info, verbose=False)

    times = preproc_raw.times.copy()
    return mode_raw, valid_mask, times


# ------------------------------------------------------------------
# Feature regressors on the alpha timeline
# ------------------------------------------------------------------
def get_event_onsets_seconds(subject, meg_data, epoch_id):
    """Event onset times in seconds (0-based, same time base as the alpha timeline).

    Reuses this project's event-definition logic so buttons, fixations,
    saccades, DA events, etc. are defined exactly as in the evoked / TRF
    analyses. ``define_events`` returns event sample indices that include the
    raw's first_samp, so we subtract ``first_time`` to get 0-based seconds.
    """
    _, events, _, _ = functions_analysis.define_events(subject=subject, meg_data=meg_data,
                                                       epoch_id=epoch_id)
    if events is None or len(events) == 0:
        return np.array([], dtype=float)
    sfreq = meg_data.info["sfreq"]
    return np.asarray(events[:, 0] / sfreq - meg_data.first_time, dtype=float)


def bad_annotations_from_mask(valid_mask, sfreq, description="BAD_gap"):
    """Turn a per-sample validity mask into MNE 'BAD' annotations.

    Contiguous runs where ``valid_mask < 0.5`` (trimmed / bad-segment gaps) are
    annotated so that ``mne.Epochs(..., reject_by_annotation=True)`` drops any
    epoch overlapping a gap. 0-based onsets (RawArray first_samp == 0).
    """
    invalid = np.asarray(valid_mask) < 0.5
    n = len(invalid)
    if not np.any(invalid):
        return mne.Annotations([], [], [])

    edges = np.diff(invalid.astype(int))
    starts = np.where(edges == 1)[0] + 1
    stops = np.where(edges == -1)[0] + 1
    if invalid[0]:
        starts = np.r_[0, starts]
    if invalid[-1]:
        stops = np.r_[stops, n]

    onsets = starts / sfreq
    durations = (stops - starts) / sfreq
    return mne.Annotations(onset=onsets, duration=durations,
                           description=[description] * len(onsets))


def _is_continuous(feature):
    return any(key in feature for key in _CONTINUOUS_KEYS)


def _continuous_feature_array(feature, subject, mode_times):
    """Build a continuous regressor on the alpha (250 Hz) timeline.

    Mirrors the continuous-feature handling in functions_analysis.make_mtrf_input
    (derivative / z-score / min-max), then interpolates the processed-raw signal
    onto the 250 Hz timeline.
    """
    raw = load.meg(subject_id=subject.subject_id, meg_params={"data_type": "processed"})
    src_times = raw.times  # 0-based seconds at the processed sfreq

    if "audio_env" in feature:
        pick = "AudioEnvVideo" if "AudioEnvVideo" in raw.ch_names else None
        if pick is None:
            raise ValueError(
                f"'AudioEnvVideo' no está en el raw de {subject.subject_id}; "
                f"corré preprocess_audio.py primero.")
        sig = raw.get_data(picks=pick)[0, :]
    else:
        feature_name = feature.replace("_std", "").replace("_der", "")
        sig = raw.get_data(picks=feature_name)[0, :]

    if "_der" in feature:
        sig = np.gradient(sig)
    if "_std" in feature:
        sig = (sig - np.mean(sig)) / np.std(sig)
    elif "_norm" in feature:
        sig = (sig - np.min(sig)) / (np.max(sig) - np.min(sig))

    # Resample (linear interpolation) onto the alpha timeline
    return np.interp(mode_times, src_times, sig)


def _event_feature_array(feature, subject, mode_times):
    """Build an impulse regressor (1 at each event onset) on the alpha timeline.

    Uses functions_analysis.define_events so onsets match the evoked / TRF
    analyses exactly (fixations, saccades, pursuits, buttons, DA events, ...).
    Onsets are converted to 0-based seconds, then to alpha sample indices.
    """
    meg_data = load.meg(subject_id=subject.subject_id, meg_params={"data_type": "processed"})
    onset_sec = get_event_onsets_seconds(subject, meg_data, feature)

    arr = np.zeros(len(mode_times), dtype=np.float64)
    if len(onset_sec) == 0:
        yprint(f">>> Sin eventos para '{feature}' en {subject.subject_id}")
        return arr

    idx = np.round(onset_sec * FS_ALPHA).astype(int)
    idx = idx[(idx >= 0) & (idx < len(arr))]
    arr[idx] = 1.0
    return arr


def make_mode_trf_input(feature, subject, mode_times, valid_mask):
    """Return a single feature regressor on the alpha timeline, gap-masked."""
    if _is_continuous(feature):
        arr = _continuous_feature_array(feature, subject, mode_times)
    else:
        arr = _event_feature_array(feature, subject, mode_times)
    # Zero-out trimmed / bad-segment samples (same idea as bad_annotations_array)
    return arr * valid_mask


# ------------------------------------------------------------------
# Statistics: 1-D temporal cluster permutation (per mode)
# ------------------------------------------------------------------
def temporal_cluster_test(data, t_thresh=None, n_permutations=1024,
                          pval_threshold=0.05, seed=42):
    """One-sample temporal cluster permutation test on a single mode curve.

    Parameters
    ----------
    data : ndarray, shape (n_subjects, n_times)
    t_thresh : dict (TFCE, e.g. dict(start=0, step=0.2)) or float

    Returns
    -------
    sig_mask : ndarray of bool, shape (n_times,)
        True where the across-subjects response differs significantly from 0.
    """
    from mne.stats import permutation_cluster_1samp_test

    if t_thresh is None:
        t_thresh = dict(start=0, step=0.2)

    out_type = "indices" if isinstance(t_thresh, dict) else "mask"
    _, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
        X=data, threshold=t_thresh, n_permutations=n_permutations,
        adjacency=None, out_type=out_type, seed=seed, n_jobs=1, verbose=False)

    n_times = data.shape[1]
    sig_mask = np.zeros(n_times, dtype=bool)

    if isinstance(t_thresh, dict):
        # TFCE: cluster_pv is a per-time-point p-value array
        sig_mask = np.asarray(cluster_pv).reshape(n_times) < pval_threshold
    else:
        for cl, pv in zip(clusters, cluster_pv):
            if pv < pval_threshold:
                sig_mask[cl[0]] = True
    return sig_mask


