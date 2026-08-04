# -*- coding: utf-8 -*-
"""
Compare two ways of getting a source-space evoked response on a surface
parcellation:

  Pipeline A  (continuous -> source -> epoch -> average)
      1. beamform the CONTINUOUS raw data  (apply_lcmv_raw)
      2. epoch the continuous source timecourses around the event
      3. average the source epochs  -> evoked_A

  Pipeline B  (sensor evoked -> source)
      1. epoch + average the SENSOR data      -> evoked_sensor
      2. beamform the sensor evoked            (apply_lcmv) -> evoked_B

  Pipeline C  (TRF on continuous source)
      1. beamform the CONTINUOUS raw data  (apply_lcmv_raw)
      2. fit a single-impulse TRF (same feature) on the source timecourses
      3. read the TRF kernel                   -> evoked_C
      C is linear like A, but DECONVOLVES overlapping events instead of
      superimposing them, so it equals A only up to that overlap correction.

If the beamformer were fully linear these are identical
    average( W @ X_raw )  ==  W @ average( X_raw ).
With pick_ori=None the beamformer returns the free-orientation MAGNITUDE
||x,y,z||, a NONLINEAR (rectifying) step that does NOT commute with averaging,
so A and B differ. This module quantifies and visualizes that difference.

Config mirrors source_parcellation_trf.py so the same forward / LCMV files are
reused. Surface parcellations only.

Outputs (per subject and grand-average):
  * GFP overlay of evoked_A vs evoked_B (+ difference metrics printed/CSV)
  * per-source time-course overlays for the most responsive sources
  * side-by-side brain plots of A and B at selected latencies
"""

import os
import mne
import numpy as np
import mne.beamformer as beamformer
import matplotlib.pyplot as plt

import functions_analysis
import functions_general
import load
import setup
import paths
import save
import plot_general


# ============================ Config ============================ #
exp_info = setup.exp_info()

# --------- Save / Display ---------#
use_saved_data = True
save_data = True
save_fig = True
display_figs = True
plot_individuals = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

# --------- MEG / source params (keep aligned with source_parcellation_trf.py) --------- #
meg_params = {'chs_id': 'mag_z',
              'band_id': None,
              'data_type': 'processed',
              'filter_sensors': True}

surf_vol = 'parcellation'      # surface parcellation only
parc = 'aparc.a2009s'
pick_ori = None                # free-orientation magnitude (the nonlinear case)
spacing = 'ico4'

# --------- Event / epoch params --------- #
feature = 'fix'                # fixations
tmin, tmax = -0.2, 0.5
baseline = (tmin, tmax)        # whole-window baseline (matches old TRF convention)
plot_margin = 0.15             # seconds cropped from each side of plotted time series

# Latencies (s) for side-by-side brain plots; None -> time of peak activation
compare_latencies = [0.0, 0.1, 0.2]
# Number of most-responsive sources to overlay as time courses
n_top_sources = 3

# --------- Pipeline C: TRF on continuous source timecourses --------- #
# A single-impulse TRF (same feature) fit on raw_src. This is linear like A,
# but DECONVOLVES overlapping fixations instead of superimposing them, so it is
# "equivalent to A" only up to that overlap correction.
trf_alpha = 0          # 0 = no ridge penalty (closest to a plain deconvolution)
trf_standarize = False  # keep source units comparable to A / B

subjects_dir = os.path.join(paths.mri_path, 'freesurfer')
os.environ["SUBJECTS_DIR"] = subjects_dir

parc_tag = f'parcellation_{parc}'

# --------- Figure / save paths --------- #
# NOTE: only strip ':' from the dynamic sub-path, NOT from the whole path —
# otherwise the drive letter ('D:') loses its colon and becomes a relative
# folder, dumping all outputs under Scripts\D\... instead of D:\...
_dyn_subpath = (f"Source_Evoked_Comparison_{meg_params['data_type']}/Band_{meg_params['band_id']}/"
                f"{parc_tag}/{feature}_{tmin}_{tmax}_bline{baseline}_ori{pick_ori}/"
                f"{meg_params['chs_id']}/").replace(":", "")
fig_path = paths.plots_path + _dyn_subpath
save_path = paths.save_path + _dyn_subpath


# ============================ Helpers ============================ #
def build_surface_label_names_positions(stc, src, subject_code):
    """Return (label_names, label_positions) aligned to stc rows.

    Names come from the parcellation annotation; positions are the surface-RAS
    coordinates of each centroid vertex. Alignment uses stc.vertices (the
    beamformer's actual output), which may drop a duplicate centroid.
    """
    labels = mne.read_labels_from_annot(subject_code, parc=parc, subjects_dir=subjects_dir)
    lh_verts = stc.vertices[0]
    rh_verts = stc.vertices[1]

    label_names = []
    for hemi_idx, (verts, hemi) in enumerate(((lh_verts, 'lh'), (rh_verts, 'rh'))):
        for vert in verts:
            matched = False
            for label in labels:
                if label.hemi == hemi and vert in label.vertices:
                    label_names.append(label.name)
                    matched = True
                    break
            if not matched:
                label_names.append(f'{hemi}_vert{vert}')

    lh_pos = src[0]['rr'][lh_verts]
    rh_pos = src[1]['rr'][rh_verts]
    all_pos = np.vstack([lh_pos, rh_pos])
    return label_names, all_pos


def comparison_metrics(data_a, data_b):
    """Per-source correlation and relative max difference between A and B."""
    n_t = min(data_a.shape[1], data_b.shape[1])
    a, b = data_a[:, :n_t], data_b[:, :n_t]
    per_src_corr = np.array([
        np.corrcoef(a[i], b[i])[0, 1]
        if np.std(a[i]) > 0 and np.std(b[i]) > 0 else np.nan
        for i in range(a.shape[0])])
    denom = max(np.abs(a).max(), 1e-30)
    rel_max_diff = np.abs(a - b).max() / denom
    overall_corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    return per_src_corr, overall_corr, rel_max_diff


# Line styles per pipeline for overlay plots
_PIPE_STYLE = {'A': dict(ls='-', label='A: continuous -> epoch -> average'),
               'B': dict(ls='--', label='B: sensor evoked -> source'),
               'C': dict(ls=':', label='C: TRF on continuous source')}


def plot_gfp_overlay(evokeds, title, out_path, fname):
    """Overlay the GFP (std across sources) of each pipeline.

    evokeds : dict {tag: Evoked} with tag in {'A','B','C'}.
    """
    ref = next(iter(evokeds.values()))
    n_t = min(ev.data.shape[1] for ev in evokeds.values())
    t = ref.times[:n_t]
    m = (t >= tmin + plot_margin) & (t <= tmax - plot_margin)
    fig, ax = plt.subplots(figsize=(8, 4))
    for tag, ev in evokeds.items():
        gfp = ev.data[:, :n_t].std(axis=0)
        st = _PIPE_STYLE.get(tag, dict(ls='-', label=tag))
        ax.plot(t[m], gfp[m], st['ls'], label=st['label'])
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('time (s)'); ax.set_ylabel('GFP')
    ax.set_title(title); ax.legend(); fig.tight_layout()
    if save_fig:
        os.makedirs(out_path, exist_ok=True)
        fig.savefig(out_path + fname + '.png', dpi=120)
    if not display_figs:
        plt.close(fig)


def plot_top_source_overlays(evokeds, ref_tag, title, out_path, fname):
    """Overlay time courses of the n_top_sources most-responsive sources.

    evokeds : dict {tag: Evoked}. Top sources are chosen from evokeds[ref_tag].
    """
    n_t = min(ev.data.shape[1] for ev in evokeds.values())
    ref = evokeds[ref_tag]
    t = ref.times[:n_t]
    m = (t >= tmin + plot_margin) & (t <= tmax - plot_margin)
    top = np.argsort(np.abs(ref.data[:, :n_t]).max(axis=1))[::-1][:n_top_sources]
    fig, axes = plt.subplots(len(top), 1, figsize=(8, 2.4 * len(top)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, top):
        for tag, ev in evokeds.items():
            st = _PIPE_STYLE.get(tag, dict(ls='-'))
            ax.plot(t[m], ev.data[s, :n_t][m], st['ls'], label=tag)
        ax.axvline(0, color='k', lw=0.5)
        ax.set_title(f'{ref.ch_names[s]}', fontsize=9)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel('time (s)')
    fig.suptitle(title)
    fig.tight_layout()
    if save_fig:
        os.makedirs(out_path, exist_ok=True)
        fig.savefig(out_path + fname + '.png', dpi=120)
    if not display_figs:
        plt.close(fig)


def evoked_to_full_stc(evoked, subject_code):
    """Expand a parcel-named Evoked into a full-surface SourceEstimate.

    Each vertex within a label gets that label's value, producing a proper
    brain-plottable / morphable stc (uniform coloring per region).
    """
    return functions_analysis.evoked_to_parcellation_stc(
        evoked, parc, subject_code, subjects_dir, spacing)


def plot_stc_brains(stc, src_full, subject_code, force_fsaverage, out_path, fname_prefix):
    """Brain plots of a source estimate at each compare latency."""
    for it in compare_latencies:
        it_suffix = f'_{it}s' if it is not None else ''
        plot_general.sources(
            stc=stc, src=src_full, subject=subject_code, subjects_dir=subjects_dir,
            initial_time=it, surf_vol='surface', force_fsaverage=force_fsaverage,
            source_estimation='trf', views=['lateral', 'medial'],
            plot_margin=plot_margin, save_fig=save_fig, fig_path=out_path,
            fname=f'{fname_prefix}{it_suffix}')


# ============================ Run ============================ #
evokeds_A = []   # continuous -> epoch -> average
evokeds_B = []   # sensor evoked -> source
evokeds_C = []   # TRF on continuous source timecourses
stcs_A_fs = []   # per-subject full-surface stc (pipeline A), morphed to fsaverage
stcs_B_fs = []   # per-subject full-surface stc (pipeline B), morphed to fsaverage
stcs_C_fs = []   # per-subject full-surface stc (pipeline C), morphed to fsaverage
metric_rows = []

# fsaverage full-surface source space (ico4) — morph target AND plotting src, so
# the morphed stcs live on exactly the vertices we later plot against.
fname_src_full_ga = paths.sources_path + f'fsaverage/fsaverage_surface_{spacing}-src.fif'
src_full_ga = mne.read_source_spaces(fname_src_full_ga)

for subject_id in exp_info.subjects_ids:

    print(f"\n{'='*60}\nSubject {subject_id}\n{'='*60}")
    subject = setup.subject(subject_id=subject_id)

    fs_subj_path = os.path.join(subjects_dir, subject_id)
    try:
        subject_code = subject_id if len(os.listdir(fs_subj_path)) else 'fsaverage'
    except Exception:
        subject_code = 'fsaverage'

    # --------- Load MEG --------- #
    meg_data = load.meg(subject_id=subject_id, meg_params=meg_params)
    picks = functions_general.pick_chs(chs_id=meg_params['chs_id'], info=meg_data.info)
    meg_data.pick(picks)
    meg_data.info.normalize_proj()
    sfreq = meg_data.info['sfreq']

    # --------- Forward --------- #
    sources_path_subject = paths.sources_path + subject_id
    fname_fwd = (sources_path_subject +
                 f'/{subject_code}_{meg_params["data_type"]}_chs{meg_params["chs_id"]}_{parc_tag}-fwd.fif')
    fwd = mne.read_forward_solution(fname_fwd)
    fwd.pick_channels(meg_data.ch_names)
    src = fwd['src']

    # --------- LCMV filters (reuse the TRF script's file) --------- #
    fname_lcmv = (sources_path_subject +
                  f'/{subject_code}_{meg_params["data_type"]}_chs{meg_params["chs_id"]}'
                  f'_band{meg_params["band_id"]}_{parc_tag}_{pick_ori}-lcmv.h5')
    if os.path.isfile(fname_lcmv) and use_saved_data:
        filters = mne.beamformer.read_beamformer(fname_lcmv)
    else:
        data_cov = mne.compute_raw_covariance(meg_data)
        filters = beamformer.make_lcmv(info=meg_data.info, forward=fwd, data_cov=data_cov,
                                       reg=0.05, pick_ori=pick_ori)
        if save_data:
            filters.save(fname=fname_lcmv, overwrite=True)

    # --------- Continuous source reconstruction --------- #
    stc_cont = beamformer.apply_lcmv_raw(meg_data, filters)
    source_data = stc_cont.data
    n_src = source_data.shape[0]

    # Label names + positions aligned to stc rows
    label_names, all_pos = build_surface_label_names_positions(stc_cont, src, subject_code)
    info_src = mne.create_info(label_names, sfreq, ch_types='misc')
    raw_src = mne.io.RawArray(source_data, info_src)
    # Positions keyed by the (de-duplicated) channel names
    label_positions = {name: all_pos[i] for i, name in enumerate(raw_src.ch_names)}

    # --------- Events --------- #
    _, events, event_id, _ = functions_analysis.define_events(
        subject=subject, meg_data=meg_data, epoch_id=feature)
    first_samp = int(round(meg_data.first_time * sfreq))
    n_pre = int(round(-tmin * sfreq))
    n_post = int(round(tmax * sfreq))
    print(f'{len(events)} {feature} events found')

    # --------- Pipeline B: sensor evoked -> source --------- #
    # (epoch the sensor data first so both pipelines use the SAME surviving trials)
    ep_sensor = mne.Epochs(meg_data, events, event_id={feature: 1}, tmin=tmin, tmax=tmax,
                           baseline=baseline, preload=True, reject=None, proj=True,
                           reject_by_annotation=True, event_repeated='drop')
    sel = ep_sensor.selection
    evoked_sensor = ep_sensor.average()
    stc_from_evoked = beamformer.apply_lcmv(evoked_sensor, filters)
    evoked_B = mne.EvokedArray(stc_from_evoked.data, info_src.copy(), tmin=tmin,
                               nave=len(ep_sensor), comment='B_sensor_evoked_to_source')

    # --------- Pipeline A: continuous source -> epoch -> average --------- #
    events_keep = events[sel].copy()
    events_keep[:, 0] = events_keep[:, 0] - first_samp
    fit_win = (events_keep[:, 0] - n_pre >= 0) & (events_keep[:, 0] + n_post < raw_src.n_times)
    events_keep = events_keep[fit_win]
    ep_src = mne.Epochs(raw_src, events_keep, event_id={feature: 1}, tmin=tmin, tmax=tmax,
                        baseline=baseline, preload=True, reject=None, proj=False,
                        reject_by_annotation=False, event_repeated='drop')
    evoked_A = ep_src.average(picks='all')
    evoked_A.comment = 'A_continuous_epoch_average'
    print(f'  trials: sensor(B)={len(ep_sensor)}, source(A)={len(ep_src)}')

    # --------- Pipeline C: TRF on continuous source timecourses --------- #
    # Build the fixation impulse-train predictor (same feature) and fit a
    # single-feature TRF on raw_src, then read the kernel as evoked_C.
    subj_trf_path = paths.save_path + f'TRF/{subject.subject_id}/'
    fname_bad_annot = 'bad_annot_array.pkl'
    if os.path.exists(subj_trf_path + fname_bad_annot) and use_saved_data:
        bad_annotations_array = load.var(subj_trf_path + fname_bad_annot)
    else:
        bad_annotations_array = functions_analysis.get_bad_annot_array(
            meg_data=meg_data, subj_path=subj_trf_path, fname=fname_bad_annot, save_var=False)
    input_arrays = functions_analysis.make_mtrf_input(
        input_arrays={}, var_name=feature, subject=subject, meg_data=meg_data,
        bad_annotations_array=bad_annotations_array, subj_path=subj_trf_path,
        fname=f'{feature}_array.pkl', save_var=False)
    model_input = input_arrays[feature][:, np.newaxis]   # (n_times, 1)

    rf = functions_analysis.fit_mtrf(
        meg_data=raw_src, tmin=tmin, tmax=tmax, alpha=trf_alpha,
        model_input=model_input, chs_id='misc',
        standarize=trf_standarize, fit_power=False)
    trf_kernel = rf.coef_[:, 0, :]                       # (n_src, n_lags)
    evoked_C = mne.EvokedArray(trf_kernel, info_src.copy(), tmin=tmin,
                               nave=len(ep_src), comment='C_source_TRF')
    # Same whole-window baseline as A/B for a like-for-like comparison
    evoked_C.apply_baseline(baseline)

    # --------- Comparison metrics (A vs B, C vs A, C vs B) --------- #
    per_AB, corr_AB, reldiff_AB = comparison_metrics(evoked_A.data, evoked_B.data)
    per_CA, corr_CA, reldiff_CA = comparison_metrics(evoked_C.data, evoked_A.data)
    per_CB, corr_CB, reldiff_CB = comparison_metrics(evoked_C.data, evoked_B.data)
    print(f'  corr A-B={corr_AB:.3f} (med psrc {np.nanmedian(per_AB):.3f}) | '
          f'C-A={corr_CA:.3f} (med psrc {np.nanmedian(per_CA):.3f}) | '
          f'C-B={corr_CB:.3f} (med psrc {np.nanmedian(per_CB):.3f})')
    metric_rows.append({'subject': subject_id,
                        'corr_AB': corr_AB, 'medcorr_AB': float(np.nanmedian(per_AB)),
                        'corr_CA': corr_CA, 'medcorr_CA': float(np.nanmedian(per_CA)),
                        'corr_CB': corr_CB, 'medcorr_CB': float(np.nanmedian(per_CB)),
                        'n_trials': len(ep_src)})

    evokeds_A.append(evoked_A)
    evokeds_B.append(evoked_B)
    evokeds_C.append(evoked_C)

    # --------- Full-surface source estimates (brain-plottable objects) --------- #
    stc_A, src_full = evoked_to_full_stc(evoked_A, subject_code)
    stc_B, _ = evoked_to_full_stc(evoked_B, subject_code)
    stc_C, _ = evoked_to_full_stc(evoked_C, subject_code)

    subj_fig_path = fig_path + f'{subject_id}/'
    subj_save_path = save_path + f'{subject_id}/'
    if save_data:
        os.makedirs(subj_save_path, exist_ok=True)
        stc_A.save(subj_save_path + f'{feature}_A', overwrite=True)
        stc_B.save(subj_save_path + f'{feature}_B', overwrite=True)
        stc_C.save(subj_save_path + f'{feature}_C', overwrite=True)

    # Morph each subject's stc to fsaverage for a geometrically-correct GA.
    # src_to pins the target vertices to the ico4 fsaverage source space so the
    # morphed stc matches src_full_ga used for plotting.
    if subject_code != 'fsaverage':
        morph = mne.compute_source_morph(src_full, subject_from=subject_code,
                                         subject_to='fsaverage', src_to=src_full_ga,
                                         subjects_dir=subjects_dir)
        stc_A_fs = morph.apply(stc_A)
        stc_B_fs = morph.apply(stc_B)
        stc_C_fs = morph.apply(stc_C)
    else:
        stc_A_fs, stc_B_fs, stc_C_fs = stc_A, stc_B, stc_C
    stcs_A_fs.append(stc_A_fs)
    stcs_B_fs.append(stc_B_fs)
    stcs_C_fs.append(stc_C_fs)

    # --------- Per-subject figures --------- #
    if plot_individuals:
        trio = {'A': evoked_A, 'B': evoked_B, 'C': evoked_C}
        plot_gfp_overlay(trio, f'GFP A/B/C - {subject_id}',
                         subj_fig_path, f'{feature}_gfp_overlay')
        plot_top_source_overlays(trio, 'A', f'Top sources A/B/C - {subject_id}',
                                 subj_fig_path, f'{feature}_top_sources')
        # spatial-colored evoked plots for each pipeline
        for ev, tag in ((evoked_A, 'A'), (evoked_B, 'B'), (evoked_C, 'C')):
            plot_general.plot_source_evoked_spatial(
                evoked=ev, label_positions=label_positions, gfp=True,
                xlim=(tmin + plot_margin, tmax - plot_margin),
                title=f'{feature} pipeline {tag} - {subject_id}',
                display_figs=display_figs, save_fig=save_fig,
                fig_path=subj_fig_path, fname=f'{feature}_evoked_{tag}')
        # per-participant brain plots on the subject's own surface
        plot_stc_brains(stc_A, src_full, subject_code, False, subj_fig_path, f'{feature}_brain_A')
        plot_stc_brains(stc_B, src_full, subject_code, False, subj_fig_path, f'{feature}_brain_B')
        plot_stc_brains(stc_C, src_full, subject_code, False, subj_fig_path, f'{feature}_brain_C')


# ============================ Grand average ============================ #
print(f"\n{'='*60}\nGrand average\n{'='*60}")

ga_A = mne.grand_average(evokeds_A, interpolate_bads=True)
ga_B = mne.grand_average(evokeds_B, interpolate_bads=True)
ga_C = mne.grand_average(evokeds_C, interpolate_bads=True)

# GA label positions on fsaverage (for spatial-colored plot)
ga_labels = mne.read_labels_from_annot('fsaverage', parc=parc, subjects_dir=subjects_dir)
fname_src_ga = paths.sources_path + f'fsaverage/fsaverage_surface_{spacing}-src.fif'
ga_label_positions = {}
if os.path.isfile(fname_src_ga):
    src_ga_space = mne.read_source_spaces(fname_src_ga)
    lh_v, rh_v = src_ga_space[0]['vertno'], src_ga_space[1]['vertno']
    all_pos_ga = np.vstack([src_ga_space[0]['rr'][lh_v], src_ga_space[1]['rr'][rh_v]])
    ga_names = []
    for verts, hemi in ((lh_v, 'lh'), (rh_v, 'rh')):
        for vert in verts:
            matched = False
            for label in ga_labels:
                if label.hemi == hemi and vert in label.vertices:
                    ga_names.append(label.name); matched = True; break
            if not matched:
                ga_names.append(f'{hemi}_vert{vert}')
    for i, name in enumerate(ga_names):
        ga_label_positions[name] = all_pos_ga[i]

# GA metrics
per_AB_ga, corr_AB_ga, _ = comparison_metrics(ga_A.data, ga_B.data)
per_CA_ga, corr_CA_ga, _ = comparison_metrics(ga_C.data, ga_A.data)
per_CB_ga, corr_CB_ga, _ = comparison_metrics(ga_C.data, ga_B.data)
print(f'  GA corr A-B={corr_AB_ga:.3f} (med psrc {np.nanmedian(per_AB_ga):.3f}) | '
      f'C-A={corr_CA_ga:.3f} (med psrc {np.nanmedian(per_CA_ga):.3f}) | '
      f'C-B={corr_CB_ga:.3f} (med psrc {np.nanmedian(per_CB_ga):.3f})')
metric_rows.append({'subject': 'GA',
                    'corr_AB': corr_AB_ga, 'medcorr_AB': float(np.nanmedian(per_AB_ga)),
                    'corr_CA': corr_CA_ga, 'medcorr_CA': float(np.nanmedian(per_CA_ga)),
                    'corr_CB': corr_CB_ga, 'medcorr_CB': float(np.nanmedian(per_CB_ga)),
                    'n_trials': np.nan})

# GA figures
ga_trio = {'A': ga_A, 'B': ga_B, 'C': ga_C}
plot_gfp_overlay(ga_trio, 'GFP A/B/C - GA', fig_path, f'{feature}_GA_gfp_overlay')
plot_top_source_overlays(ga_trio, 'A', 'Top sources A/B/C - GA', fig_path, f'{feature}_GA_top_sources')
for ev, tag in ((ga_A, 'A'), (ga_B, 'B'), (ga_C, 'C')):
    plot_general.plot_source_evoked_spatial(
        evoked=ev, label_positions=ga_label_positions if ga_label_positions else None,
        gfp=True, xlim=(tmin + plot_margin, tmax - plot_margin),
        title=f'{feature} pipeline {tag} - GA',
        display_figs=display_figs, save_fig=save_fig,
        fig_path=fig_path, fname=f'{feature}_GA_evoked_{tag}')

# --------- GA brain plots from morphed source estimates (geometric average) --------- #
# Average the fsaverage-morphed stcs vertex-by-vertex, then render on fsaverage.
ga_stc_A = stcs_A_fs[0].copy()
ga_stc_A.data = np.mean([s.data for s in stcs_A_fs], axis=0)
ga_stc_A.subject = 'fsaverage'
ga_stc_B = stcs_B_fs[0].copy()
ga_stc_B.data = np.mean([s.data for s in stcs_B_fs], axis=0)
ga_stc_B.subject = 'fsaverage'
ga_stc_C = stcs_C_fs[0].copy()
ga_stc_C.data = np.mean([s.data for s in stcs_C_fs], axis=0)
ga_stc_C.subject = 'fsaverage'

# ============================ Save (before plotting, so results persist
# even if the 3D brain rendering fails) ============================ #
import pandas as pd
if save_data:
    ga_stc_A.save(save_path + f'{feature}_GA_A', overwrite=True)
    ga_stc_B.save(save_path + f'{feature}_GA_B', overwrite=True)
    ga_stc_C.save(save_path + f'{feature}_GA_C', overwrite=True)
    save.var(var={'evokeds_A': evokeds_A, 'evokeds_B': evokeds_B, 'evokeds_C': evokeds_C,
                  'ga_A': ga_A, 'ga_B': ga_B, 'ga_C': ga_C,
                  'ga_stc_A': ga_stc_A, 'ga_stc_B': ga_stc_B, 'ga_stc_C': ga_stc_C},
             path=save_path, fname='source_evoked_comparison.pkl')

os.makedirs(fig_path, exist_ok=True)
pd.DataFrame(metric_rows).to_csv(fig_path + 'comparison_metrics.csv', index=False)
print(f'Saved comparison metrics: {fig_path}comparison_metrics.csv')

# --------- GA brain plots from morphed source estimates (geometric average) --------- #
# Average the fsaverage-morphed stcs vertex-by-vertex, then render on fsaverage.
plot_stc_brains(ga_stc_A, src_full_ga, 'fsaverage', True, fig_path, f'{feature}_GA_brain_A')
plot_stc_brains(ga_stc_B, src_full_ga, 'fsaverage', True, fig_path, f'{feature}_GA_brain_B')
plot_stc_brains(ga_stc_C, src_full_ga, 'fsaverage', True, fig_path, f'{feature}_GA_brain_C')

print('Done.')

