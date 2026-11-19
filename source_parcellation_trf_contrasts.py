# -*- coding: utf-8 -*-
"""
Post-hoc phase contrasts for the source-parcellation TRF.

The source TRF is fitted with the phase pieces only (fix_CF / fix_DA / fix_Audio,
same for sac). Here the decomposition into a common response and phase effects is
made after the fit, per subject, and tested across subjects:

    common          = (CF + DA + Audio) / 3           (simple mean)
    <phase>-common  = phase effect beyond the common response
    DA-CF, Audio-CF, Audio-DA                          pairwise differences

Each contrast is treated like a feature: grand average, spatial-coloured evoked
plot, cluster permutation test across regions/time, masked brain of the
significant regions, and a CSV summary. Point this script at the run folder via
RUN_SUBDIR (relative to Save/ and Plots/).
"""
import os
import re
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

import paths
import load
import save
import functions_analysis
import plot_general

# --------- Run to analyse ---------#
UNITS = 'standardized'   # 'standardized' | 'original' (kernel units folder of the source TRF run)
RUN_SUBDIR = ("TRF_Source_processed/Band_(0.1, 40)/parcellation_aparc.a2009s/"
              "{fix_CF, fix_DA, fix_Audio, sac_CF, sac_DA, sac_Audio, "
              "Steering_std_der, Gas_std_der, Brake_std_der, left_but, right_but}"
              f"_-0.2_0.5_trflearn_median_oriNone/mag_z/{UNITS}/")
parc = 'aparc.a2009s'
spacing = 'ico4'
BASES = ('fix', 'sac')
PHASES = ('CF', 'DA', 'Audio')

# --------- Statistics / plots (same settings as source_parcellation_trf.py) ---------#
pval_threshold = 0.05
t_thresh = dict(start=0, step=0.2)
n_permutations = 1024
plot_margin = 0.15
initial_time = {'fix': [None], 'sac': [0.0, 0.12]}   # brain snapshots per base
save_fig = True
display_figs = False
plt.ioff()

save_path = os.path.join(paths.save_path, RUN_SUBDIR)
fig_path = os.path.join(paths.plots_path, RUN_SUBDIR, 'contrasts/')
os.makedirs(fig_path, exist_ok=True)
subjects_dir = os.path.join(paths.mri_path, 'freesurfer')

# --------- Load per-subject evokeds ---------#
feature_evokeds = load.var(os.path.join(save_path, 'GA_all_subjects.pkl'))
# Subject IDs per feature (DA1 has no Audio phase, so the *_Audio features hold DA2 only)
feature_subjects = load.var(os.path.join(save_path, 'GA_subjects.pkl'))
by_subject = {feature: dict(zip(feature_subjects[feature], evokeds)) for feature, evokeds in feature_evokeds.items()}
print(f'>>> features: {list(feature_evokeds)}; subjects per feature: '
      f'{ {f: len(s) for f, s in feature_subjects.items()} }')


def contrast_evokeds(base):
    """Per-subject contrast evokeds for one base feature -> {contrast_name: [evoked, ...]}.

    Subjects are paired by ID. Each contrast takes the subjects that have the phases it needs:
    common and <phase>-common need all three phases; pairwise differences need both phases.
    """
    out = {}
    subjects = sorted(set().union(*[by_subject[f'{base}_{p}'] for p in PHASES]))
    for s in subjects:
        ev = {p: by_subject[f'{base}_{p}'][s] for p in PHASES if s in by_subject[f'{base}_{p}']}
        # Restrict to regions present in all of this subject's phases
        chs = [ch for ch in next(iter(ev.values())).ch_names if all(ch in e.ch_names for e in ev.values())]
        ev = {p: e.copy().pick(chs) for p, e in ev.items()}
        contrasts = {}
        if len(ev) == len(PHASES):
            common = ev['CF'].copy()
            common.data = np.mean([ev[p].data for p in PHASES], axis=0)
            contrasts[f'{base}_common'] = common
            for p in PHASES:
                contrasts[f'{base}_{p}-common'] = mne.combine_evoked([ev[p], common], weights=[1, -1])
        for a, b in (('DA', 'CF'), ('Audio', 'CF'), ('Audio', 'DA')):
            if a in ev and b in ev:
                contrasts[f'{base}_{a}-{b}'] = mne.combine_evoked([ev[a], ev[b]], weights=[1, -1])
        for name, c in contrasts.items():
            c.comment = name
            out.setdefault(name, []).append(c)
    return out


# --------- Label centroids on fsaverage for spatial colouring ---------#
labels = mne.read_labels_from_annot('fsaverage', parc=parc, subjects_dir=subjects_dir)
centroid = {lab.name: lab.pos.mean(axis=0) for lab in labels}


def positions_for(ch_names):
    # channel names may carry a "-N" duplicate suffix
    return {ch: centroid[re.sub(r'-\d+$', '', ch)] for ch in ch_names if re.sub(r'-\d+$', '', ch) in centroid}


# --------- Run ---------#
sig_rows = []
clusters = {}
for base in BASES:
    if not all(f'{base}_{p}' in feature_evokeds for p in PHASES):
        print(f'>>> {base}: missing phase features, skipping')
        continue
    contrasts = contrast_evokeds(base)
    tmin, tmax = next(iter(contrasts.values()))[0].times[[0, -1]]

    # regions common to every subject of every contrast, adjacency once per base
    all_evs = [ev for evs in contrasts.values() for ev in evs]
    common_chs = set.intersection(*[set(ev.ch_names) for ev in all_evs])
    stat_regions = [ch for ch in all_evs[0].ch_names if ch in common_chs]
    label_positions = positions_for(stat_regions)
    adjacency = functions_analysis.get_parcellation_adjacency(
        ch_names=stat_regions, surf_vol='parcellation', subject='fsaverage',
        subjects_dir=subjects_dir, parc=parc, label_positions=label_positions)

    for name, evs in contrasts.items():
        print(f'\n>>> {name} (N={len(evs)} subjects)')
        evs = [ev.copy().pick(stat_regions) for ev in evs]
        ga = mne.grand_average(evs, interpolate_bads=False)

        plot_general.plot_source_evoked_spatial(
            evoked=ga, label_positions=label_positions, gfp=True,
            xlim=(tmin + plot_margin, tmax - plot_margin), title=f'GA - {name}',
            display_figs=display_figs, save_fig=save_fig, fig_path=fig_path, fname=f'{name}_GA_source_trf')

        data = np.array([ev.data.T for ev in evs])          # (subjects, times, regions)
        mask_tr, pvals = functions_analysis.run_permutations_test(
            data=data, pval_threshold=pval_threshold, t_thresh=t_thresh,
            adj_matrix=adjacency, n_permutations=n_permutations)
        mask_rt = mask_tr.T
        clusters[name] = {'mask': mask_rt, 'pvalues': pvals}

        for r_idx, region in enumerate(stat_regions):
            sig_times = ga.times[mask_rt[r_idx]]
            if sig_times.size:
                sig_rows.append({'contrast': name, 'region': region, 'n_sig_times': int(sig_times.size),
                                 't_start': float(sig_times.min()), 't_end': float(sig_times.max())})

        if mask_rt.any():
            ga_sig = ga.copy()
            ga_sig.data = ga.data * mask_rt
            stc_sig, src_sig = functions_analysis.evoked_to_parcellation_stc(
                ga_sig, parc, 'fsaverage', subjects_dir, spacing)
            for it in initial_time[base]:
                it_suffix = f'_{it}s' if it is not None else ''
                plot_general.sources(
                    stc=stc_sig, src=src_sig, subject='fsaverage', subjects_dir=subjects_dir,
                    initial_time=it, surf_vol='surface', force_fsaverage=True,
                    source_estimation='trf', views=['lateral', 'medial'], plot_margin=plot_margin,
                    save_fig=save_fig, fig_path=fig_path, fname=f'{name}_GA_source_trf_sig{it_suffix}')
            print(f'    {int(mask_rt.any(axis=1).sum())} significant regions')
        else:
            print(f'    no significant clusters (p < {pval_threshold})')
        plt.close('all')

save.var(var={'clusters': clusters, 'pval_threshold': pval_threshold, 't_thresh': t_thresh,
              'n_permutations': n_permutations}, path=save_path, fname='GA_stats_contrasts.pkl')
pd.DataFrame(sig_rows, columns=['contrast', 'region', 'n_sig_times', 't_start', 't_end']) \
    .to_csv(os.path.join(fig_path, 'significant_regions_contrasts.csv'), index=False)
print(f'\n>>> Done. Figures in {fig_path}')
