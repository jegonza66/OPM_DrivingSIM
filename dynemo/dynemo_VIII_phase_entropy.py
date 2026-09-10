"""Entropy of the DyNeMo mode profile across experiment phases (CF / DA / Audiobook).

Window-profile metrics adapted from Türker et al. (2023, Sci Rep 13:21260).
Each WINDOW_SECONDS window is summarised by the time average of the per-sample
normalised mixing coefficients: a probability distribution over modes whose
Shannon entropy says how many modes shared the window and with what weight.
(By concavity this is >= the mean instantaneous entropy: half a window in mode
3 then half in mode 5 gives 1 bit even if every sample was pure.)

A. Intra-subject entropy. Per subject and phase, mean over windows. Fixed-length
   windows remove the phase-duration confound (longer spans visit more modes).
   Unit of inference: subject. Friedman across phases + paired Wilcoxon (Holm).
   The mean mode profiles per phase are reported alongside to show which
   redistribution of modes any entropy change comes from; the profile itself
   needs no windows and is taken over every usable sample of the phase.

B. Inter-subject divergence. Only the Audiobook is a stimulus shared across
   subjects (CF / DA are each subject's own drive), so windows there are tiled
   from the audio onset and window k covers the same audio for everyone.
   H(group mean profile) - mean H(profile) is the generalised Jensen-Shannon
   divergence between subjects. It is tested against a null that circularly
   shifts each subject's window sequence: every within-subject statistic is kept
   and only the alignment is destroyed. Lower observed than null = convergence.

C. Hard dominance, over each subject's whole phase series (no windows needed).
   Intra-subject occupancy: share of samples in which each mode is the argmax,
   per subject and phase (Friedman + paired Wilcoxon, Holm within mode).
   Group modal share: subjects aligned to the phase onset; at each sample the
   mode most subjects are in, then the share of the phase each mode held that
   plurality. Tested per phase against the same circular-shift null, so it asks
   whether a mode's dominance exceeds what its base rate alone would give.
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
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import paths
import setup
import load
import functions_analysis
from general_utility_functions import cprint, yprint, rprint
import dynemo__mixing_coefficients_utils as mc
from dynemo_config import (n_modes as N_MODES, n_pca as N_PCA, n_embeddings as N_EMBEDDINGS,
                           sequence_length as SEQUENCE_LENGTH, ch_picks as CH_PICKS)

################ SETUP ################
use_reweighted_alpha = True
WINDOW_SECONDS = 1.0
PHASES = ["CF", "DA", "Audio"]
ALIGNED_PHASE = "Audio"        # the one phase with a stimulus shared across subjects
MIN_SUBJECTS_PER_WINDOW = 5    # for the inter-subject measures (7 subjects in total)
N_SURROGATES = 1000
RANDOM_STATE = 42

# tab10, same as plot_alpha_stack in stage V
mode_colors = [f"C{i}" for i in range(10)]
phase_colors = {"CF": "tab:gray", "DA": "tab:orange", "Audio": "tab:blue"}

infered_parameters_path = paths.dynemo_run_save_path(
    N_MODES, N_PCA, N_EMBEDDINGS, SEQUENCE_LENGTH, CH_PICKS, "DyNeMo_Infered_Parameters")
alpha_subdir = "alpha_reweighted" if use_reweighted_alpha else "alpha"
SAVE_PATH = paths.dynemo_run_save_path(
    N_MODES, N_PCA, N_EMBEDDINGS, SEQUENCE_LENGTH, CH_PICKS,
    os.path.join("DyNeMo_Phase_Entropy", alpha_subdir, f"win{WINDOW_SECONDS:g}s"))
PLOT_PATH = paths.dynemo_run_plots_path(
    N_MODES, N_PCA, N_EMBEDDINGS, SEQUENCE_LENGTH, CH_PICKS,
    os.path.join("Phase_Entropy", alpha_subdir, f"win{WINDOW_SECONDS:g}s"))


################ HELPERS ################
def _entropy(p, axis=-1):
    """Shannon entropy in bits; zeros contribute nothing, NaN propagates."""
    safe = np.where(p > 0, p, 1.0)
    return -(p * np.log2(safe)).sum(axis=axis)


def _holm(pvalues):
    """Holm step-down adjusted p values."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(
        1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    return adjusted


def _subject_phase_data(subject_code, alpha_i):
    """Window table (one row per phase, window) plus whole-phase summaries.

    The second return maps phase -> {"labels", "profile"}: the argmax label
    series from the phase onset at the alpha rate (-1 outside the phase or in
    trimmed / bad-segment gaps) and the mean normalised alpha over usable samples.
    """
    mode_raw, valid_mask, mode_times = mc.build_mode_raw(
        subject_code=subject_code, alpha_i=alpha_i, ch_picks=CH_PICKS,
        n_pca=N_PCA, n_embeddings=N_EMBEDDINGS, sequence_length=SEQUENCE_LENGTH)
    data = mode_raw.get_data(picks="misc")
    win = int(round(WINDOW_SECONDS * mode_raw.info["sfreq"]))
    valid = valid_mask > 0.5

    meg_data = load.meg(subject_id=subject_code, meg_params={"data_type": "processed"})
    phase_masks = functions_analysis.get_experiment_phase_mask(subject_code, meg_data)

    rows = []
    whole = {}
    for phase in PHASES:
        # onto the alpha timeline, same convention as the continuous regressors
        mask = np.interp(mode_times, meg_data.times, phase_masks[phase].astype(float)) > 0.5
        if not mask.any():
            yprint(f">>> {subject_code}: sin fase {phase}")
            continue
        on = np.flatnonzero(mask)
        usable = mask & valid
        series = np.where(usable, data.argmax(axis=0), -1)[on[0]:on[-1] + 1]
        kept = data[:, usable]
        whole[phase] = {
            "labels": series.astype(np.int8),
            "profile": (kept / kept.sum(axis=0, keepdims=True)).mean(axis=1),
        }
        # tile from the phase onset: for the shared stimulus, window k is the
        # same audio for every subject
        for k, first in enumerate(range(on[0], on[-1] - win + 2, win)):
            block = slice(first, first + win)
            if not usable[block].all():
                continue
            seg = data[:, block]
            # composition of the mixture, not overall alpha magnitude
            profile = (seg / seg.sum(axis=0, keepdims=True)).mean(axis=1)
            row = {"subject": subject_code, "phase": phase, "window": k,
                   "onset": mode_times[first], "entropy": _entropy(profile)}
            row.update({f"mode_{m + 1}": profile[m] for m in range(len(profile))})
            rows.append(row)
    return pd.DataFrame(rows), whole


def _paired_tests(summary, value):
    """Friedman across phases + pairwise Wilcoxon (Holm) on subjects with all phases."""
    wide = summary.pivot(index="subject", columns="phase", values=value)[PHASES].dropna()
    friedman_p = stats.friedmanchisquare(*[wide[p] for p in PHASES]).pvalue
    pairs = [(a, b) for i, a in enumerate(PHASES) for b in PHASES[i + 1:]]
    raw = [stats.wilcoxon(wide[a], wide[b]).pvalue for a, b in pairs]
    return wide, friedman_p, dict(zip(pairs, _holm(raw)))


def _jsd_per_window(profiles):
    """Generalised JSD between subjects per window; NaN where too few subjects.

    profiles : (n_subjects, n_windows, n_modes) with NaN for missing windows.
    """
    n_present = (~np.isnan(profiles[..., 0])).sum(axis=0)
    with np.errstate(all="ignore"):
        group = np.nanmean(profiles, axis=0)
        mean_entropy = np.nanmean(_entropy(profiles), axis=0)
    jsd = _entropy(group) - mean_entropy
    jsd[n_present < MIN_SUBJECTS_PER_WINDOW] = np.nan
    return jsd


def _stack_labels(series_list):
    """Right-pad label series with -1 into one (n_subjects, n_samples) array."""
    longest = max(len(s) for s in series_list)
    out = np.full((len(series_list), longest), -1, dtype=np.int8)
    for i, s in enumerate(series_list):
        out[i, :len(s)] = s
    return out


def _group_modal_share(labels, n_modes):
    """Share of aligned samples in which each mode is the plurality across subjects.

    labels : (n_subjects, n_samples) argmax per subject, -1 where missing.
    Samples with fewer than MIN_SUBJECTS_PER_WINDOW subjects are ignored.
    """
    counts = np.stack([(labels == m).sum(axis=0) for m in range(n_modes)])
    enough = counts.sum(axis=0) >= MIN_SUBJECTS_PER_WINDOW
    if not enough.any():
        return np.full(n_modes, np.nan)
    return np.bincount(counts.argmax(axis=0)[enough], minlength=n_modes) / enough.sum()


def _shift_each(labels, rng):
    """Circularly shift each subject's valid labels in place of themselves.

    Gaps stay where they are, so which samples reach MIN_SUBJECTS_PER_WINDOW is
    identical for observed and null; only the temporal order is destroyed.
    """
    shifted = labels.copy()
    for row in shifted:
        idx = np.flatnonzero(row >= 0)
        if len(idx) >= 2:
            row[idx] = np.roll(row[idx], rng.integers(1, len(idx)))
    return shifted


################ PLOTS ################
def _plot_paired(wide, friedman_p, pair_p, output_file):
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(PHASES))
    for _, row in wide.iterrows():
        ax.plot(x, row[PHASES], color="gray", alpha=0.4, linewidth=1)
    ax.plot(x, wide[PHASES].mean(), color="tab:red", linewidth=2.5, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(PHASES)
    ax.set_xlim(-0.3, len(PHASES) - 0.7)
    ax.set_ylabel("Window-profile entropy (bits)")

    values = wide[PHASES].to_numpy()
    top, span = values.max(), values.max() - values.min()
    for level, ((a, b), p) in enumerate(pair_p.items()):
        y = top + span * (0.08 + 0.14 * level)
        i, j = PHASES.index(a), PHASES.index(b)
        ax.plot([i, i, j, j], [y, y + 0.02 * span, y + 0.02 * span, y],
                color="black", linewidth=1)
        ax.text((i + j) / 2, y + 0.03 * span, f"{mc.stars(p)} p={p:.2g}",
                ha="center", va="bottom", fontsize="small")
    ax.set_ylim(top=top + span * (0.08 + 0.14 * len(pair_p) + 0.1))
    ax.set_title(f"Intra-subject entropy (N={len(wide)}), Friedman p={friedman_p:.2g}",
                 fontsize="medium")
    fig.tight_layout()
    mc.save_figure(fig, output_file)
    plt.close(fig)


def _plot_profiles(phase_profiles, mode_p, ylabel, title, output_file):
    """phase_profiles[phase] is (n_subjects, n_modes); stars from mode_p per mode."""
    n_modes = next(iter(phase_profiles.values())).shape[1]
    modes = np.arange(1, n_modes + 1)
    fig, ax = plt.subplots(figsize=(1.6 * n_modes, 5))
    width = 0.8 / len(PHASES)
    for i, phase in enumerate(PHASES):
        values = phase_profiles[phase]
        sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
        ax.bar(modes + (i - (len(PHASES) - 1) / 2) * width, values.mean(axis=0), width,
               yerr=sem, capsize=3, label=phase, color=phase_colors[phase], alpha=0.85)
    top = max(v.mean(axis=0).max() for v in phase_profiles.values())
    for m in range(n_modes):
        ax.text(m + 1, top * 1.05, mc.stars(mode_p[m]), ha="center")
    ax.set_xticks(modes)
    ax.set_xlabel("DyNeMo mode")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize="medium")
    ax.legend()
    fig.tight_layout()
    mc.save_figure(fig, output_file)
    plt.close(fig)


def _plot_group_modal(observed, null, pvalues, output_file):
    """Observed share per mode and phase, null mean as a black tick, stars vs null."""
    n_modes = len(next(iter(observed.values())))
    modes = np.arange(1, n_modes + 1)
    fig, ax = plt.subplots(figsize=(1.6 * n_modes, 5))
    width = 0.8 / len(PHASES)
    for i, phase in enumerate(PHASES):
        if phase not in observed:
            continue
        x = modes + (i - (len(PHASES) - 1) / 2) * width
        ax.bar(x, observed[phase], width, label=phase, color=phase_colors[phase], alpha=0.85)
        ax.hlines(null[phase].mean(axis=0), x - width / 2, x + width / 2,
                  color="black", linewidth=1.5)
        for m in range(n_modes):
            ax.text(x[m], observed[phase][m] + 0.01, mc.stars(pvalues[phase][m]),
                    ha="center", fontsize="small")
    ax.plot([], [], color="black", linewidth=1.5, label="shifted null mean")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_xticks(modes)
    ax.set_xlabel("DyNeMo mode")
    ax.set_ylabel("Share of phase as group modal mode")
    ax.set_title("Group plurality per phase (stars: > shifted null, Holm over modes)",
                 fontsize="medium")
    ax.legend()
    fig.tight_layout()
    mc.save_figure(fig, output_file)
    plt.close(fig)


def _plot_divergence(times, jsd, null_curves, observed, null_means, p, output_file):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5),
                                      gridspec_kw={"width_ratios": [2, 1]})
    low, high = np.nanpercentile(null_curves, [2.5, 97.5], axis=0)
    left.fill_between(times, low, high, color="gray", alpha=0.25, linewidth=0,
                      label="shifted null, 95%")
    left.plot(times, np.nanmean(null_curves, axis=0), color="gray", linewidth=1)
    left.plot(times, jsd, color="tab:blue", linewidth=1.5, label="aligned")
    left.set_xlabel(f"Time from {ALIGNED_PHASE} onset (s)")
    left.set_ylabel("Inter-subject JSD (bits)")
    left.legend(frameon=False)

    right.hist(null_means, bins=30, color="gray", alpha=0.7)
    right.axvline(observed, color="tab:blue", linewidth=2)
    right.set_xlabel("Mean JSD over windows (bits)")
    right.set_ylabel("Surrogates")
    right.set_title(f"aligned < shifted: {mc.stars(p)} p={p:.2g}", fontsize="medium")
    fig.suptitle(f"Inter-subject divergence during {ALIGNED_PHASE} "
                 f"({WINDOW_SECONDS:g} s windows, >= {MIN_SUBJECTS_PER_WINDOW} subjects)")
    fig.tight_layout()
    mc.save_figure(fig, output_file)
    plt.close(fig)


################ MAIN ################
def _compare_modes(table, mode_cols, tag, stats_rows, ylabel, title, output_file):
    """Per-mode Friedman (Holm over modes) + pairwise Wilcoxon across phases, then bars."""
    phase_values = {phase: [] for phase in PHASES}
    friedman = []
    n = None
    for col in mode_cols:
        wide, fp, pp = _paired_tests(table, col)
        n = len(wide)
        friedman.append(fp)
        for phase in PHASES:
            phase_values[phase].append(wide[phase].to_numpy())
        for (a, b), p in pp.items():
            stats_rows.append({"measure": f"{tag}_{col}", "unit": "subject", "n": n,
                               "test": f"wilcoxon {a} vs {b} (holm within mode)", "p": p})
    adjusted = _holm(friedman)
    for col, raw, adj in zip(mode_cols, friedman, adjusted):
        stats_rows.append({"measure": f"{tag}_{col}", "unit": "subject", "n": n,
                           "test": "friedman (holm over modes)", "p": adj, "p_raw": raw})
    _plot_profiles({phase: np.column_stack(v) for phase, v in phase_values.items()},
                   adjusted, ylabel, title, output_file)


def main():
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(PLOT_PATH, exist_ok=True)

    alp = mc.load_alpha(use_reweighted=use_reweighted_alpha,
                        infered_parameters_path=infered_parameters_path)
    subjects = setup.exp_info().subjects_ids
    cprint(f">>> Sujetos en alpha: {len(alp)}; en lista: {len(subjects)}")

    frames = []
    labels = {phase: {} for phase in PHASES}
    profile_rows = []
    for i, subject_code in enumerate(subjects):
        if i >= len(alp):
            rprint(f">>> No hay alpha para {subject_code}")
            continue
        table, whole = _subject_phase_data(subject_code, alp[i])
        for phase, item in whole.items():
            labels[phase][subject_code] = item["labels"]
            profile_rows.append({"subject": subject_code, "phase": phase,
                                 "n_samples": int((item["labels"] >= 0).sum()),
                                 **{f"mode_{m + 1}": v for m, v in enumerate(item["profile"])}})
        counts = table.groupby("phase").size().reindex(PHASES, fill_value=0)
        cprint(f">>> {subject_code}: ventanas " +
               ", ".join(f"{phase}={n}" for phase, n in counts.items()))
        frames.append(table)
    if not frames:
        rprint(">>> Ningún sujeto con alphas utilizables")
        return

    profiles = pd.concat(frames, ignore_index=True)
    profiles.to_csv(os.path.join(SAVE_PATH, "window_profiles.csv"), index=False)
    mode_cols = [c for c in profiles.columns if c.startswith("mode_")]

    grouped = profiles.groupby(["subject", "phase"])
    summary = grouped[["entropy"]].mean()
    summary["n_windows"] = grouped.size()
    summary = summary.reset_index()
    summary.to_csv(os.path.join(SAVE_PATH, "subject_phase_entropy.csv"), index=False)

    phase_profiles = pd.DataFrame(profile_rows)
    phase_profiles.to_csv(os.path.join(SAVE_PATH, "subject_phase_profiles.csv"), index=False)

    stats_rows = []

    ########## A. intra-subject entropy ##########
    wide, friedman_p, pair_p = _paired_tests(summary, "entropy")
    if len(wide) < summary["subject"].nunique():
        yprint(f">>> {summary['subject'].nunique() - len(wide)} sujetos sin las "
               f"{len(PHASES)} fases, excluidos de los tests pareados")
    stats_rows.append({"measure": "entropy", "unit": "subject", "n": len(wide),
                       "test": "friedman", "p": friedman_p})
    for (a, b), p in pair_p.items():
        stats_rows.append({"measure": "entropy", "unit": "subject", "n": len(wide),
                           "test": f"wilcoxon {a} vs {b} (holm)", "p": p})
    cprint(f">>> Entropía intra-sujeto: Friedman p={friedman_p:.3g}; " +
           ", ".join(f"{a} vs {b} p={p:.3g}" for (a, b), p in pair_p.items()))
    _plot_paired(wide, friedman_p, pair_p,
                 os.path.join(PLOT_PATH, "phase_entropy_paired.png"))

    ########## mode profiles (soft) and occupancy (hard), whole phase, unit = subject ##########
    _compare_modes(phase_profiles, mode_cols, "profile", stats_rows,
                   "Mean alpha share over phase",
                   "Mode profiles per phase (stars: Friedman, Holm over modes)",
                   os.path.join(PLOT_PATH, "phase_mode_profiles.png"))

    occupancy_rows = []
    for phase, per_subject in labels.items():
        for code, s in per_subject.items():
            held = s[s >= 0]
            if len(held) == 0:
                continue
            share = np.bincount(held, minlength=len(mode_cols)) / len(held)
            occupancy_rows.append({"subject": code, "phase": phase, "n_samples": len(held),
                                   **dict(zip(mode_cols, share))})
    occupancy = pd.DataFrame(occupancy_rows)
    occupancy.to_csv(os.path.join(SAVE_PATH, "subject_phase_occupancy.csv"), index=False)
    _compare_modes(occupancy, mode_cols, "occupancy", stats_rows,
                   "Share of samples as subject's argmax mode",
                   "Hard occupancy per phase (stars: Friedman, Holm over modes)",
                   os.path.join(PLOT_PATH, "phase_mode_occupancy.png"))

    ########## group modal share vs shifted null, per phase ##########
    rng = np.random.default_rng(RANDOM_STATE)
    modal_obs, modal_null, modal_p = {}, {}, {}
    for phase, per_subject in labels.items():
        if len(per_subject) < MIN_SUBJECTS_PER_WINDOW:
            yprint(f">>> {phase}: solo {len(per_subject)} sujetos, sin dominancia de grupo")
            continue
        stacked = _stack_labels(list(per_subject.values()))
        observed = _group_modal_share(stacked, len(mode_cols))
        null = np.stack([_group_modal_share(_shift_each(stacked, rng), len(mode_cols))
                         for _ in range(N_SURROGATES)])
        raw = (np.sum(null >= observed, axis=0) + 1) / (N_SURROGATES + 1)
        modal_obs[phase], modal_null[phase], modal_p[phase] = observed, null, _holm(raw)
        for col, obs, mean_null, p_raw, p_adj in zip(mode_cols, observed, null.mean(axis=0),
                                                     raw, modal_p[phase]):
            stats_rows.append({"measure": f"group_modal_{col}", "unit": "sample",
                               "n": len(per_subject), "phase": phase,
                               "test": f"circular-shift surrogates x{N_SURROGATES}, observed > null (holm over modes)",
                               "p": p_adj, "p_raw": p_raw, "observed": obs, "null_mean": mean_null})
        cprint(f">>> Modo modal de grupo en {phase}: " +
               ", ".join(f"{c}={o:.3f} (nulo {n:.3f}, p={p:.3g})"
                         for c, o, n, p in zip(mode_cols, observed, null.mean(axis=0), modal_p[phase])))
    if modal_obs:
        pd.DataFrame({f"{phase}_{k}": v for phase in modal_obs
                      for k, v in (("observed", modal_obs[phase]),
                                   ("null_mean", modal_null[phase].mean(axis=0)),
                                   ("p_holm", modal_p[phase]))},
                     index=mode_cols).to_csv(os.path.join(SAVE_PATH, "group_modal_share.csv"))
        _plot_group_modal(modal_obs, modal_null, modal_p,
                          os.path.join(PLOT_PATH, "phase_group_modal_mode.png"))

    ########## B. inter-subject divergence on the shared stimulus ##########
    aligned = profiles[profiles["phase"] == ALIGNED_PHASE]
    codes = sorted(aligned["subject"].unique())
    if len(codes) >= MIN_SUBJECTS_PER_WINDOW:
        n_windows = int(aligned["window"].max()) + 1
        matrix = np.full((len(codes), n_windows, len(mode_cols)), np.nan)
        for s, code in enumerate(codes):
            rows = aligned[aligned["subject"] == code]
            matrix[s, rows["window"].to_numpy(int)] = rows[mode_cols].to_numpy()

        jsd = _jsd_per_window(matrix)
        observed = np.nanmean(jsd)
        rng = np.random.default_rng(RANDOM_STATE)
        null_curves = np.empty((N_SURROGATES, n_windows))
        for i in range(N_SURROGATES):
            shifted = matrix.copy()
            for s in range(len(codes)):
                # roll only the subject's present windows, keeping its gaps fixed
                idx = np.flatnonzero(~np.isnan(matrix[s, :, 0]))
                if len(idx) >= 2:
                    shifted[s, idx] = np.roll(matrix[s, idx], rng.integers(1, len(idx)), axis=0)
            null_curves[i] = _jsd_per_window(shifted)
        null_means = np.nanmean(null_curves, axis=1)
        p = (np.sum(null_means <= observed) + 1) / (N_SURROGATES + 1)
        times = np.arange(n_windows) * WINDOW_SECONDS

        pd.DataFrame({"time": times, "jsd_aligned": jsd,
                      "jsd_null_mean": np.nanmean(null_curves, axis=0),
                      "n_subjects": (~np.isnan(matrix[..., 0])).sum(axis=0)}).to_csv(
            os.path.join(SAVE_PATH, f"{ALIGNED_PHASE}_intersubject_jsd.csv"), index=False)
        stats_rows.append({"measure": "intersubject_jsd", "unit": "window",
                           "n": int(np.isfinite(jsd).sum()),
                           "test": f"circular-shift surrogates x{N_SURROGATES}, aligned < shifted",
                           "p": p, "observed": observed, "null_mean": null_means.mean()})
        cprint(f">>> JSD inter-sujeto ({ALIGNED_PHASE}): {observed:.4f} bits vs nulo "
               f"{null_means.mean():.4f}, p={p:.3g}")
        _plot_divergence(times, jsd, null_curves, observed, null_means, p,
                         os.path.join(PLOT_PATH, f"{ALIGNED_PHASE}_intersubject_jsd.png"))
    else:
        yprint(f">>> Solo {len(codes)} sujetos con ventanas en {ALIGNED_PHASE}; "
               f"se omite la divergencia inter-sujeto")

    pd.DataFrame(stats_rows).to_csv(os.path.join(SAVE_PATH, "phase_stats.csv"), index=False)
    cprint(f">>> Resultados en {SAVE_PATH}")
    cprint(">>> Análisis de entropía por fase terminado.")


if __name__ == "__main__":
    main()
