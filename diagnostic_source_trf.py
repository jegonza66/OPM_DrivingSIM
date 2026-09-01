# -*- coding: utf-8 -*-
"""
Diagnostic: why does the source-space TRF not show the fixation evoked
response that source-estimating the evoked does?

Because the LCMV beamformer is a LINEAR, time-invariant operator W
(n_sources x n_channels), the following are mathematically identical:

    mean_fix( W @ X_raw )  ==  W @ mean_fix( X_raw )  ==  apply_lcmv(evoked)

i.e. epoching the *continuous* source reconstruction around fixations and
averaging == beamforming the sensor-evoked (the pipeline that works).
So the continuous reconstruction cannot be discarding the fixation response.

This script localizes the problem with two tests on ONE subject:

  Test A (reconstruction fidelity):
      epoch raw_src around fixations + average   vs   apply_lcmv(evoked_sensor)
      -> must match to numerical precision. Exonerates the projection.

  Test B (TRF vs simple average on the SAME source timecourses):
      fix TRF kernel   vs   source epoch-average
      with standarize on/off and a pre-event baseline.
      -> if the average shows the response but the TRF does not, the cause is
         the TRF model (overlap deconvolution / standardization / baseline),
         NOT the source reconstruction.

Mirrors the setup of source_parcellation_trf.py so paths/filters are reused.
"""

import os
import mne
import numpy as np
import mne.beamformer as beamformer
import matplotlib
matplotlib.use('Agg')          # non-interactive: save figures, never block
import matplotlib.pyplot as plt

import functions_analysis
import functions_general
import load
import setup
import paths

# --------- Config (keep aligned with source_parcellation_trf.py) ---------#
meg_params = {'chs_id': 'mag_z',
              'band_id': None,
              'data_type': 'processed',
              'filter_sensors': True}

surf_vol = 'parcellation'      # 'parcellation' | 'vol_parcellation'
parc = 'aparc.a2009s'
vol_parc_name = 'aal'          # only used for path tag if surf_vol == 'vol_parcellation'
pick_ori = None
spacing = 'ico4'
pos = 10

feature = 'fix'
tmin, tmax = -0.2, 0.5
baseline = (tmin, -0.05)        # pre-event baseline (NOT the whole window)

subjects_dir = os.path.join(paths.mri_path, 'freesurfer')
os.environ["SUBJECTS_DIR"] = subjects_dir

if surf_vol == 'vol_parcellation':
    parc_tag = f'vol_parcellation_{vol_parc_name}'
else:
    parc_tag = f'parcellation_{parc}'

# --------- Pick a subject ---------#
exp_info = setup.exp_info()
subject_id = exp_info.subjects_ids[0]
subject = setup.subject(subject_id=subject_id)
print(f'Diagnostic on subject {subject_id}')

fs_subj_path = os.path.join(subjects_dir, subject_id)
try:
    subject_code = subject_id if len(os.listdir(fs_subj_path)) else 'fsaverage'
except Exception:
    subject_code = 'fsaverage'

# --------- Load MEG + forward + LCMV filters (same as TRF script) ---------#
meg_data = load.meg(subject_id=subject_id, meg_params=meg_params)
picks = functions_general.pick_chs(chs_id=meg_params['chs_id'], info=meg_data.info)
meg_data.pick(picks)
meg_data.info.normalize_proj()

sources_path_subject = paths.sources_path + subject_id
fname_fwd = (sources_path_subject +
             f'/{subject_code}_{meg_params["data_type"]}_chs{meg_params["chs_id"]}_{parc_tag}-fwd.fif')
fwd = mne.read_forward_solution(fname_fwd)
fwd.pick_channels(meg_data.ch_names)

fname_lcmv = (sources_path_subject +
              f'/{subject_code}_{meg_params["data_type"]}_chs{meg_params["chs_id"]}'
              f'_band{meg_params["band_id"]}_{parc_tag}_{pick_ori}-lcmv.h5')
if os.path.isfile(fname_lcmv):
    filters = mne.beamformer.read_beamformer(fname_lcmv)
else:
    data_cov = mne.compute_raw_covariance(meg_data)
    filters = beamformer.make_lcmv(info=meg_data.info, forward=fwd, data_cov=data_cov,
                                   reg=0.05, pick_ori=pick_ori)

# --------- Continuous source reconstruction ---------#
stc_cont = beamformer.apply_lcmv_raw(meg_data, filters)     # (n_src, n_times)
source_data = stc_cont.data
n_src = source_data.shape[0]
sfreq = meg_data.info['sfreq']

# Wrap continuous source data as a misc Raw so we can epoch it with MNE
src_ch_names = [f'src{i}' for i in range(n_src)]
info_src = mne.create_info(src_ch_names, sfreq, ch_types='misc')
raw_src = mne.io.RawArray(source_data, info_src)

# --------- Fixation events ---------#
metadata, events, event_id, _ = functions_analysis.define_events(
    subject=subject, meg_data=meg_data, epoch_id=feature)

# Sample index relative to the start of the recording (raw_src has first_samp=0)
first_samp = int(round(meg_data.first_time * sfreq))
n_pre = int(round(-tmin * sfreq))
n_post = int(round(tmax * sfreq))
print(f'{len(events)} fixation events found')

# =====================================================================
# TEST A: reconstruction fidelity
#   epoch-average of continuous source  ==  apply_lcmv(evoked_sensor) ?
#
# CRITICAL: both averages must use the EXACT same set of trials. The sensor
# epochs drop bad/annotated epochs (and apply SSP); the source RawArray has no
# annotations/projectors and would drop none. We therefore epoch the sensor
# data first, take the surviving selection, and build the source epochs from
# exactly those events so the comparison is a pure linearity check.
# =====================================================================
print('\n=== TEST A: continuous-source epoch-average vs beamformed evoked ===')

# (1) epoch + average the SENSOR data (real rejection + SSP), then beamform
ep_sensor = mne.Epochs(meg_data, events, event_id={feature: 1}, tmin=tmin, tmax=tmax,
                       baseline=baseline, preload=True, reject=None, proj=True,
                       reject_by_annotation=True, event_repeated='drop')
sel = ep_sensor.selection                      # indices into `events` that survived
evoked_sensor = ep_sensor.average()
stc_from_evoked = beamformer.apply_lcmv(evoked_sensor, filters)

# (2) epoch + average the CONTINUOUS source reconstruction over the SAME trials
events_keep = events[sel].copy()
events_keep[:, 0] = events_keep[:, 0] - first_samp
fit_win = (events_keep[:, 0] - n_pre >= 0) & (events_keep[:, 0] + n_post < raw_src.n_times)
events_keep = events_keep[fit_win]
ep_src = mne.Epochs(raw_src, events_keep, event_id={feature: 1}, tmin=tmin, tmax=tmax,
                    baseline=baseline, preload=True, reject=None, proj=False,
                    reject_by_annotation=False, event_repeated='drop')
evoked_src_epochavg = ep_src.average(picks='all')   # (n_src, n_times)
print(f'  trials: sensor={len(ep_sensor)}, source={len(ep_src)}')

a = evoked_src_epochavg.data
b = stc_from_evoked.data
# Match time length defensively
n_t = min(a.shape[1], b.shape[1])
a, b = a[:, :n_t], b[:, :n_t]
diff = np.abs(a - b)
denom = np.maximum(np.abs(a).max(), 1e-30)
print(f'  max |epochavg - beamformed_evoked|      = {diff.max():.3e}')
print(f'  relative max diff (vs signal amplitude)  = {diff.max() / denom:.3e}')
corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
print(f'  overall correlation                      = {corr:.6f}')
print('  -> if ~identical (corr~1, rel diff ~1e-6), the projection is fine.')

# Output folder for diagnostic figures
diag_fig_path = paths.plots_path + 'Diagnostic_Source_TRF/'
os.makedirs(diag_fig_path, exist_ok=True)

# Visual fidelity check on the most-responsive source
peak_a = int(np.argmax(np.abs(a).max(axis=1)))
t_a = evoked_src_epochavg.times[:n_t]
plt.figure(figsize=(8, 4))
plt.plot(t_a, a[peak_a], label='continuous-source epoch-average')
plt.plot(t_a, b[peak_a], '--', label='apply_lcmv(evoked_sensor)')
plt.axvline(0, color='k', lw=0.5)
plt.title(f'TEST A fidelity - {subject_id} src#{peak_a}')
plt.xlabel('time (s)'); plt.legend(); plt.tight_layout()
plt.savefig(f'{diag_fig_path}{subject_id}_testA_fidelity.png', dpi=120)
plt.close()

# =====================================================================
# TEST B: TRF kernel vs source epoch-average (same source timecourses)
# =====================================================================
print('\n=== TEST B: fix TRF kernel vs source epoch-average ===')

# Build the fixation impulse-train input (same logic as the TRF pipeline)
subj_path = paths.save_path + f'TRF/{subject.subject_id}/'
fname_bad_annot = 'bad_annot_array.pkl'
if os.path.exists(subj_path + fname_bad_annot):
    bad_annotations_array = load.var(subj_path + fname_bad_annot)
else:
    bad_annotations_array = functions_analysis.get_bad_annot_array(
        meg_data=meg_data, subj_path=subj_path, fname=fname_bad_annot, save_var=False)

input_arrays = functions_analysis.make_mtrf_input(
    input_arrays={}, var_name=feature, subject=subject, meg_data=meg_data,
    bad_annotations_array=bad_annotations_array, subj_path=subj_path,
    fname=f'{feature}_array.pkl', save_var=False)
model_input = input_arrays[feature][:, np.newaxis]   # (n_times, 1)

# Fit two TRFs on the SAME source timecourses: standardized vs raw units
for standarize in (True, False):
    rf = functions_analysis.fit_mtrf(
        meg_data=raw_src, tmin=tmin, tmax=tmax, alpha=0,
        model_input=model_input, chs_id='misc',
        standarize=standarize, fit_power=False)
    kernel = rf.coef_[:, 0, :]                        # (n_src, n_lags)

    # Compare temporal shape against the epoch-average (per source, then mean)
    n_t2 = min(kernel.shape[1], evoked_src_epochavg.data.shape[1])
    k = kernel[:, :n_t2]
    e = evoked_src_epochavg.data[:, :n_t2]
    # correlation of each source's kernel with its epoch-average waveform
    per_src_corr = np.array([
        np.corrcoef(k[i], e[i])[0, 1] if np.std(k[i]) > 0 and np.std(e[i]) > 0 else np.nan
        for i in range(n_src)])
    print(f'  standarize={standarize}: median per-source corr(kernel, epoch-avg) '
          f'= {np.nanmedian(per_src_corr):.3f}, '
          f'frac |corr|>0.5 = {np.mean(np.abs(per_src_corr) > 0.5):.2f}')

    # Plot the most-responsive source (by epoch-average peak) for visual check
    peak_src = int(np.argmax(np.abs(e).max(axis=1)))
    t = evoked_src_epochavg.times[:n_t2]
    plt.figure(figsize=(8, 4))
    ea = e[peak_src]
    ka = k[peak_src]
    plt.plot(t, ea / (np.abs(ea).max() + 1e-30), label='epoch-average (norm.)')
    plt.plot(t, ka / (np.abs(ka).max() + 1e-30), label='TRF kernel (norm.)')
    plt.axvline(0, color='k', lw=0.5)
    plt.title(f'{subject_id} src#{peak_src}  standarize={standarize}')
    plt.xlabel('time (s)'); plt.legend(); plt.tight_layout()
    fname_fig = f'{diag_fig_path}{subject_id}_testB_kernel_vs_epochavg_std{standarize}.png'
    plt.savefig(fname_fig, dpi=120)
    plt.close()
    print(f'    saved {fname_fig}')

print('\nInterpretation:')
print('  - TEST A ~identical  -> source projection is faithful (hypothesis ruled out).')
print('  - TEST B: if epoch-avg shows the response but the kernel does not,')
print('    the cause is the TRF model: overlap deconvolution of quasi-periodic')
print('    fixations, per-channel standardization, and/or whole-window baseline.')

# =====================================================================
# TEST C: is the free-orientation magnitude (pick_ori=None) the culprit?
#   pick_ori=None on a free-orientation forward returns ||x,y,z|| -> a
#   NONLINEAR rectification that does not commute with averaging, so the
#   continuous-source epoch-average cannot equal apply_lcmv(evoked).
#   Repeat Test A with a SCALAR, sign-preserving orientation (max-power):
#   the linear identity should now hold (corr ~ 1).
# =====================================================================
print('\n=== TEST C: scalar (max-power) orientation fidelity ===')
data_cov = mne.compute_raw_covariance(meg_data, verbose='ERROR')
filters_scalar = beamformer.make_lcmv(info=meg_data.info, forward=fwd, data_cov=data_cov,
                                      reg=0.05, pick_ori='max-power')

stc_cont_s = beamformer.apply_lcmv_raw(meg_data, filters_scalar)
raw_src_s = mne.io.RawArray(stc_cont_s.data,
                            mne.create_info([f's{i}' for i in range(stc_cont_s.data.shape[0])],
                                            sfreq, ch_types='misc'), verbose='ERROR')
ep_src_s = mne.Epochs(raw_src_s, events_keep, event_id={feature: 1}, tmin=tmin, tmax=tmax,
                      baseline=baseline, preload=True, reject=None, proj=False,
                      reject_by_annotation=False, event_repeated='drop', verbose='ERROR')
a_s = ep_src_s.average(picks='all').data
b_s = beamformer.apply_lcmv(evoked_sensor, filters_scalar).data
n_ts = min(a_s.shape[1], b_s.shape[1])
a_s, b_s = a_s[:, :n_ts], b_s[:, :n_ts]
diff_s = np.abs(a_s - b_s)
corr_s = np.corrcoef(a_s.ravel(), b_s.ravel())[0, 1]
print(f'  relative max diff = {diff_s.max() / max(np.abs(a_s).max(), 1e-30):.3e}')
print(f'  overall correlation = {corr_s:.6f}')
print(f'  (pick_ori=None gave corr={corr:.3f}; scalar should be ~1.0)')
print('  -> if corr~1 here, the free-orientation magnitude (pick_ori=None) is the cause:')
print('     it rectifies the continuous source, so averaging/TRF lose the evoked polarity.')

