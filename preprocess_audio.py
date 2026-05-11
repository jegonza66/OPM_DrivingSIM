# -*- coding: utf-8 -*-
"""
Audio envelope preprocessing from driving simulator screen recording videos.

Extracts audio from video files, synchronizes with the OPM Audio channel via
cross-correlation of envelopes, computes a high-quality audio envelope from
the full-rate video audio (24 kHz), and adds it as a new channel
('AudioEnvVideo') to the MEG data.

Usage:
    - As a module: call extract_and_sync_audio(subject_id, meg_data) from
      the preprocessing pipeline
    - Standalone: run this script to process all subjects
"""

import os
import subprocess
import tempfile
import numpy as np
import scipy.signal
import scipy.io.wavfile
import mne
import paths
import setup
import load


def extract_audio_from_video(subject_id):
    """Extract audio from the screen recording video as a mono WAV array.

    Parameters
    ----------
    subject_id : str
        Participant ID.

    Returns
    -------
    audio : ndarray (n_samples,)
        Mono audio signal.
    sfreq_audio : int
        Audio sample rate (Hz).
    """
    video_path = os.path.join(paths.exp_video_path, subject_id, 'Videos',
                              f'{subject_id}_et_screenrecording.mp4')
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f'Video not found: {video_path}')

    # Extract to temporary WAV file using ffmpeg
    tmp_wav = os.path.join(tempfile.gettempdir(), f'{subject_id}_audio.wav')
    cmd = ['ffmpeg', '-y', '-i', video_path, '-vn', '-ac', '1',
           '-acodec', 'pcm_s16le', tmp_wav]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed:\n{result.stderr}')

    sfreq_audio, audio = scipy.io.wavfile.read(tmp_wav)
    audio = audio.astype(np.float64)

    # Normalize to [-1, 1]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val

    # Clean up temp file
    os.remove(tmp_wav)

    return audio, sfreq_audio


def compute_envelope(signal, sfreq, lp_freq=20.0):
    """Compute amplitude envelope via Hilbert transform + low-pass filter.

    Parameters
    ----------
    signal : ndarray (n_samples,)
        Input signal.
    sfreq : float
        Sample rate of the signal (Hz).
    lp_freq : float
        Low-pass cutoff for the envelope (Hz).

    Returns
    -------
    envelope : ndarray (n_samples,)
        Amplitude envelope.
    """
    analytic = scipy.signal.hilbert(signal)
    envelope = np.abs(analytic)
    # Low-pass filter
    b, a = scipy.signal.butter(4, lp_freq, btype='low', fs=sfreq)
    envelope = scipy.signal.filtfilt(b, a, envelope)
    return envelope


def find_sync_delay(opm_audio, sfreq_opm, video_audio, sfreq_video):
    """Find the synchronization delay between OPM and video audio.

    Computes envelopes of both signals, downsamples the video envelope to the
    OPM sample rate, and cross-correlates to find the optimal lag.

    Parameters
    ----------
    opm_audio : ndarray
        Audio signal from OPM Audio channel.
    sfreq_opm : float
        OPM sample rate (Hz).
    video_audio : ndarray
        Audio signal extracted from video.
    sfreq_video : float
        Video audio sample rate (Hz).

    Returns
    -------
    delay_samples_opm : int
        Delay in OPM samples. Positive means video audio starts after OPM
        recording (video needs to be shifted left / OPM needs offset).
    delay_seconds : float
        Delay in seconds.
    correlation : float
        Peak correlation value (for quality check).
    """
    # Compute envelopes at their native rates, using a low cutoff suitable
    # for cross-correlation at the OPM rate
    lp_freq = min(sfreq_opm / 2 - 1, 40.0)

    env_opm = compute_envelope(opm_audio, sfreq_opm, lp_freq=lp_freq)

    env_video = compute_envelope(video_audio, sfreq_video, lp_freq=lp_freq)

    # Downsample video envelope to OPM rate
    n_samples_target = int(len(env_video) * sfreq_opm / sfreq_video)
    env_video_ds = scipy.signal.resample(env_video, n_samples_target)

    # Normalize both envelopes for cross-correlation
    env_opm = (env_opm - np.mean(env_opm)) / (np.std(env_opm) + 1e-10)
    env_video_ds = (env_video_ds - np.mean(env_video_ds)) / (np.std(env_video_ds) + 1e-10)

    # Cross-correlate
    corr = scipy.signal.correlate(env_opm, env_video_ds, mode='full')
    corr /= max(len(env_opm), len(env_video_ds))
    lags = scipy.signal.correlation_lags(len(env_opm), len(env_video_ds), mode='full')

    peak_idx = np.argmax(corr)
    delay_samples_opm = lags[peak_idx]
    delay_seconds = delay_samples_opm / sfreq_opm
    correlation = corr[peak_idx]

    return delay_samples_opm, delay_seconds, correlation


def align_and_make_envelope(video_audio, sfreq_video, sfreq_opm, n_opm_samples,
                            delay_seconds, lp_freq=20.0):
    """Align video audio to OPM timeline, compute envelope, and downsample.

    Parameters
    ----------
    video_audio : ndarray
        Full-rate audio from video.
    sfreq_video : float
        Video audio sample rate (Hz).
    sfreq_opm : float
        OPM sample rate (Hz).
    n_opm_samples : int
        Number of samples in the OPM recording.
    delay_seconds : float
        Synchronization delay in seconds. Positive means video starts after
        OPM (trim video start / pad OPM start).
    lp_freq : float
        Low-pass cutoff for the envelope.

    Returns
    -------
    envelope_ds : ndarray (n_opm_samples,)
        Audio envelope aligned and downsampled to OPM timeline.
    """
    opm_duration = n_opm_samples / sfreq_opm

    # Determine where OPM t=0 falls in the video audio
    video_start_sample = int(round(-delay_seconds * sfreq_video))

    # Build aligned video audio array matching OPM duration
    n_video_samples_needed = int(round(opm_duration * sfreq_video))
    aligned = np.zeros(n_video_samples_needed)

    # Copy the relevant portion
    src_start = max(0, video_start_sample)
    dst_start = max(0, -video_start_sample)
    n_copy = min(len(video_audio) - src_start, n_video_samples_needed - dst_start)
    if n_copy > 0:
        aligned[dst_start:dst_start + n_copy] = video_audio[src_start:src_start + n_copy]

    # Compute envelope at full rate
    envelope = compute_envelope(aligned, sfreq_video, lp_freq=lp_freq)

    # Downsample to OPM rate
    envelope_ds = scipy.signal.resample(envelope, n_opm_samples)

    # Ensure non-negative (resampling can introduce small negatives)
    envelope_ds = np.maximum(envelope_ds, 0)

    return envelope_ds


def extract_and_sync_audio(subject_id, meg_data, lp_freq=20.0, save_fig=True,
                           plot=True):
    """Full pipeline: extract video audio, sync, envelope, add to meg_data.

    Parameters
    ----------
    subject_id : str
        Participant ID.
    meg_data : mne.io.Raw
        MEG data (must contain 'Audio' channel). Modified in-place.
    lp_freq : float
        Low-pass cutoff for the final audio envelope.
    save_fig : bool
        Whether to save the QC figure.
    plot : bool
        Whether to display the QC figure.

    Returns
    -------
    meg_data : mne.io.Raw
        MEG data with 'AudioEnvVideo' channel added.
    sync_info : dict
        Synchronization results (delay_seconds, delay_samples, correlation).
    """
    print(f'\n--- Audio envelope from video for subject {subject_id} ---')

    sfreq_opm = meg_data.info['sfreq']
    n_opm_samples = meg_data.n_times

    # Step 1: Extract audio from video
    print('Extracting audio from video...')
    video_audio, sfreq_video = extract_audio_from_video(subject_id)
    print(f'  Video audio: {len(video_audio)} samples at {sfreq_video} Hz '
          f'({len(video_audio)/sfreq_video:.1f} s)')

    # Step 2: Get OPM Audio channel
    opm_audio = meg_data.get_data(picks='Audio')[0, :]
    print(f'  OPM audio: {len(opm_audio)} samples at {sfreq_opm:.1f} Hz '
          f'({len(opm_audio)/sfreq_opm:.1f} s)')

    # Step 3: Cross-correlate to find sync delay
    print('Computing cross-correlation for synchronization...')
    delay_samples, delay_seconds, correlation = find_sync_delay(
        opm_audio, sfreq_opm, video_audio, sfreq_video)
    print(f'  Sync delay: {delay_seconds:.3f} s ({delay_samples} OPM samples)')
    print(f'  Peak correlation: {correlation:.4f}')

    if correlation < 0.1:
        print(f'  WARNING: Low correlation ({correlation:.4f}). '
              f'Synchronization may be unreliable!')

    # Step 4: Align and compute envelope
    print('Computing aligned audio envelope...')
    envelope_ds = align_and_make_envelope(
        video_audio, sfreq_video, sfreq_opm, n_opm_samples,
        delay_seconds, lp_freq=lp_freq)

    # Step 5: Add as new channel to meg_data
    print('Adding AudioEnvVideo channel to MEG data...')
    if 'AudioEnvVideo' in meg_data.ch_names:
        # Drop existing channel if re-running
        meg_data.drop_channels(['AudioEnvVideo'])

    info_env = mne.create_info(['AudioEnvVideo'], sfreq_opm, ch_types='misc')
    raw_env = mne.io.RawArray(envelope_ds[np.newaxis, :], info_env)
    meg_data.add_channels([raw_env], force_update_info=True)

    sync_info = {
        'delay_seconds': delay_seconds,
        'delay_samples_opm': delay_samples,
        'correlation': correlation,
        'sfreq_video': sfreq_video,
        'lp_freq': lp_freq,
    }

    # Step 6: QC plot — overlay envelopes to verify alignment
    if plot or save_fig:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

        times = meg_data.times

        # OPM envelope
        env_opm = compute_envelope(opm_audio, sfreq_opm, lp_freq=lp_freq)
        env_opm_norm = (env_opm - env_opm.min()) / (env_opm.max() - env_opm.min() + 1e-10)
        env_vid_norm = (envelope_ds - envelope_ds.min()) / (envelope_ds.max() - envelope_ds.min() + 1e-10)

        axes[0].plot(times, env_opm_norm, label='OPM Audio env', alpha=0.7)
        axes[0].plot(times, env_vid_norm, label='Video Audio env', alpha=0.7)
        axes[0].set_ylabel('Normalized envelope')
        axes[0].set_title(f'Subject {subject_id} — Audio sync '
                          f'(delay={delay_seconds:.3f}s, r={correlation:.3f})')
        axes[0].legend()

        axes[1].plot(times, env_opm_norm, label='OPM Audio env')
        axes[1].set_ylabel('OPM envelope')

        axes[2].plot(times, env_vid_norm, label='Video Audio env', color='tab:orange')
        axes[2].set_ylabel('Video envelope')
        axes[2].set_xlabel('Time (s)')

        plt.tight_layout()

        if save_fig:
            fig_dir = paths.plots_path + f'Audio_Sync/'
            os.makedirs(fig_dir, exist_ok=True)
            fig.savefig(fig_dir + f'{subject_id}_audio_sync.png', dpi=150)
            print(f'  QC figure saved to {fig_dir}')

        if plot:
            plt.show()
        else:
            plt.close(fig)

    print(f'  Done. AudioEnvVideo channel added ({n_opm_samples} samples)')
    return meg_data, sync_info


# --------- Standalone execution ---------#
if __name__ == '__main__':
    exp_info = setup.exp_info()

    for subject_id in exp_info.subjects_ids:
        print(f'\n{"="*60}')
        print(f'Processing subject {subject_id}')
        print(f'{"="*60}')

        # Load processed MEG data
        meg_params = {'data_type': 'processed'}
        meg_data = load.meg(subject_id=subject_id, meg_params=meg_params)

        # Extract, sync, and add audio envelope
        meg_data, sync_info = extract_and_sync_audio(subject_id, meg_data)

        # Save updated MEG data
        save_dir = os.path.join(paths.processed_path, subject_id)
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f'DA2_{subject_id}_meg.fif')
        meg_data.save(fname, overwrite=True)
        print(f'Saved updated data to {fname}')
