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
import save
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
def build_mode_raw(subject_code, alpha_i, ch_picks, n_pca=80, n_embeddings=15, sequence_length=100,
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
    preproc_fif = os.path.join(paths.dynemo_preprocessing_path(ch_picks), subject_code,
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
    kept_file = find_kept_times_file(subject_code, paths.dynemo_preprocessing_path(ch_picks))
    if kept_file is None:
        raise FileNotFoundError(f"No encuentro kept_times para {subject_code}")
    kept_times = np.load(kept_file)

    # Samples DyNeMo trimmed at the start (regression-spectra trimming)
    trim_start = get_subject_trim_start(subject_code=subject_code,
                                        ch_picks=ch_picks,
                                        n_modes=alpha_i.shape[1],
                                        n_pca=n_pca,
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


def _continuous_feature_array(feature, subject, mode_times, valid_mask):
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

    # Resample (linear interpolation) onto the alpha timeline
    values = np.interp(mode_times, src_times, sig)

    # Scale using only the samples that survive the gap mask: ridge is
    # scale-dependent, so trimmed / bad segments must not set the scale.
    valid = np.asarray(valid_mask) > 0.5
    if "_std" in feature:
        scale = values[valid].std()
        if scale == 0:
            raise ValueError(f"El regresor continuo '{feature}' no tiene varianza "
                             f"en {subject.subject_id}")
        values = (values - values[valid].mean()) / scale
    elif "_norm" in feature:
        minimum = values[valid].min()
        span = values[valid].max() - minimum
        if span == 0:
            raise ValueError(f"El regresor continuo '{feature}' no tiene rango "
                             f"en {subject.subject_id}")
        values = (values - minimum) / span

    return values


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
        arr = _continuous_feature_array(feature, subject, mode_times, valid_mask)
    else:
        arr = _event_feature_array(feature, subject, mode_times)
    # Zero-out trimmed / bad-segment samples (same idea as bad_annotations_array)
    return arr * valid_mask


# ------------------------------------------------------------------
# Statistics: 1-D temporal cluster permutation (per mode)
# ------------------------------------------------------------------
def temporal_cluster_test(data, t_thresh=None, n_permutations=1024,
                          pval_threshold=0.05, seed=42, return_clusters=False):
    """One-sample temporal cluster permutation test on a single mode curve.

    Parameters
    ----------
    data : ndarray, shape (n_subjects, n_times)
    t_thresh : dict or float
        dict -> TFCE, e.g. dict(start=0, step=0.2), passed straight to MNE.
        float -> two-tailed p for the cluster-forming threshold, converted to a
        t value with n_subjects - 1 df.
    return_clusters : bool
        Also return ``[(start, stop, pvalue), ...]`` with ``stop`` exclusive.
        Under TFCE a cluster is a contiguous significant run and its p value is
        the largest sample p inside it, so every sample under it meets the level.

    Returns
    -------
    sig_mask : ndarray of bool, shape (n_times,)
        True where the across-subjects response differs significantly from 0.
    """
    from mne.stats import permutation_cluster_1samp_test

    if t_thresh is None:
        t_thresh = dict(start=0, step=0.2)

    is_tfce = isinstance(t_thresh, dict)
    if is_tfce:
        threshold = t_thresh
    elif isinstance(t_thresh, float):
        from scipy import stats
        threshold = stats.t.ppf(1 - t_thresh / 2, data.shape[0] - 1)
    else:
        raise TypeError(
            "t_thresh must be a dict (TFCE) or a float (two-tailed p), "
            f"got {type(t_thresh).__name__}")

    out_type = "indices" if is_tfce else "mask"
    _, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
        X=data, threshold=threshold, n_permutations=n_permutations,
        adjacency=None, out_type=out_type, seed=seed, n_jobs=1, verbose=False)

    n_times = data.shape[1]
    sig_mask = np.zeros(n_times, dtype=bool)
    intervals = []

    if is_tfce:
        # TFCE: cluster_pv is a per-time-point p-value array
        sample_pv = np.asarray(cluster_pv).reshape(n_times)
        sig_mask = sample_pv < pval_threshold
        edges = np.flatnonzero(np.diff(np.r_[False, sig_mask, False]))
        for start, stop in zip(edges[::2], edges[1::2]):
            intervals.append((start, stop, sample_pv[start:stop].max()))
    else:
        for cl, pv in zip(clusters, cluster_pv):
            if pv < pval_threshold:
                sig_mask[cl[0]] = True
                idx = np.flatnonzero(cl[0])
                intervals.append((idx[0], idx[-1] + 1, pv))
    return (sig_mask, intervals) if return_clusters else sig_mask


def stars(pvalue):
    """Conventional significance marker for a p value."""
    for threshold, marker in ((0.001, "***"), (0.01, "**"), (0.05, "*")):
        if pvalue < threshold:
            return marker
    return "n.s."


def draw_significance_bars(axis, times, mode_clusters, colors):
    """Draw coloured cluster bars with ``stars p=`` labels above the traces.

    ``mode_clusters[mode]`` is the ``(start, stop, pvalue)`` list from
    ``temporal_cluster_test``. Each significant mode gets its own row; the
    figure grows so the data area keeps its size. Call before ``tight_layout``.
    """
    from matplotlib.font_manager import FontProperties

    active = [(mode, found) for mode, found in enumerate(mode_clusters) if found]
    if not active:
        return

    half_sample = 0.5 * np.median(np.diff(times))
    font_pt = FontProperties(size="small").get_size_in_points()
    figure = axis.figure
    axis_height_in = axis.get_position().height * figure.get_figheight()
    row_in = 1.7 * font_pt / 72
    bottom, top = axis.get_ylim()
    step = row_in * (top - bottom) / axis_height_in
    for row, (mode, found) in enumerate(active):
        y = top + step * (row + 0.35)
        color = colors[mode % len(colors)]
        edges = []
        for start, stop, _ in found:
            t0 = times[start] - half_sample
            t1 = times[stop - 1] + half_sample
            axis.hlines(y, t0, t1, color=color, linewidth=3)
            edges += [t0, t1]
        # one label per mode: the largest cluster p, centred over the mode's extent
        pvalue = max(p for _, _, p in found)
        axis.text(
            (min(edges) + max(edges)) / 2,
            y + 0.08 * step,
            f"{stars(pvalue)} p={pvalue:.2g}",
            ha="center",
            va="bottom",
            fontsize="small",
            color=color,
        )
    axis.set_ylim(bottom, top + step * (len(active) + 0.5))
    figure.set_figheight(figure.get_figheight() + row_in * (len(active) + 0.5))


def save_figure(fig, output_file, dpi=300):
    """Save a png plus an svg copy in an ``svg`` subfolder next to it."""
    path, fname = os.path.split(output_file)
    save.fig(fig, path, os.path.splitext(fname)[0], save_svg=True, dpi=dpi)


# ------------------------------------------------------------------
# Alpha time-course plots (stacked area / heatmap)
# ------------------------------------------------------------------
def _mark_events(ax, events, color="k"):
    """Vertical dashed lines with a label at the top for ``{label: time}``.

    Labels alternate between two rows so neighbouring markers do not overlap.
    """
    for i, (label, t) in enumerate(sorted((events or {}).items(), key=lambda kv: kv[1])):
        ax.axvline(t, color=color, linestyle="--", linewidth=2)
        ax.text(t, 1.02 + 0.18 * (i % 2), label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize="small", fontweight="bold")


def plot_alpha_stack(alpha, times, title=None, events=None):
    """Stacked-area plot of alpha (n_time, n_modes) on an explicit time axis."""
    import matplotlib.pyplot as plt

    n_modes = alpha.shape[1]
    colors = plt.cm.tab10.colors[:n_modes]
    fig, ax = plt.subplots(figsize=(12, 2.5), facecolor="white")
    ax.stackplot(times, alpha.T, colors=colors,
                 labels=[f"Mode {m + 1}" for m in range(n_modes)])
    ax.autoscale(tight=True)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mixing coefficient")
    ax.set_title(title, pad=30 if events else None)
    _mark_events(ax, events)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize="small",
              frameon=False)
    fig.tight_layout()
    return fig


def plot_alpha_heatmap(alpha, times, title=None, events=None, cmap="hot"):
    """Heatmap of alpha (n_time, n_modes): one row per mode, colour = activation (%).

    NaNs (e.g. trimmed / bad-segment gaps) are drawn in grey.
    """
    import matplotlib.pyplot as plt

    n_modes = alpha.shape[1]
    half_dt = 0.5 * np.median(np.diff(times))
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("lightgrey")
    fig, ax = plt.subplots(figsize=(12, 2.5), facecolor="white")
    im = ax.imshow(alpha.T * 100, aspect="auto", origin="upper", cmap=cmap,
                   vmin=0, vmax=100,
                   extent=[times[0] - half_dt, times[-1] + half_dt, n_modes + 0.5, 0.5],
                   interpolation="nearest")
    ax.set_yticks(range(1, n_modes + 1))
    ax.set_yticklabels([f"Mode {m + 1}" for m in range(n_modes)])
    ax.set_xlabel("Time (s)")
    ax.set_title(title, pad=30 if events else None)
    _mark_events(ax, events, color="cyan")  # visible on every level of the hot colormap
    fig.colorbar(im, ax=ax, pad=0.01, label="Activation (%)")
    fig.tight_layout()
    return fig


