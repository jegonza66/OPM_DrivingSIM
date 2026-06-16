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
import matplotlib.pyplot as plt
import mne

# Silence the benign Welch warning emitted for very short good spans (between
# BAD annotations), e.g. "nperseg = 1000 is greater than input length = 1".
import warnings
warnings.filterwarnings("ignore", message=".*nperseg.*greater than input length.*")

import paths
import setup
import load

from general_utility_functions import cprint, rprint, yprint, gprint


# ============================================================
# PSD CHECK - DYNEMO PREPROCESSING CHECK
# ============================================================

# Setup
exp_info = setup.exp_info()

fmin = 1
fmax = 45

# Welch window length (seconds). With reject_by_annotation, the data is split
# into contiguous good spans; a shorter, fixed window keeps the frequency
# resolution consistent and avoids most "nperseg > input length" warnings from
# short spans between BAD annotations. 4 s -> 0.25 Hz resolution.
WELCH_WIN_SEC = 4.0

# Per-subject channel/parcel-averaged spectra (1-D, length n_freqs). Used for
# the grand average across subjects, which is shape-safe even when subjects
# have different numbers of channels.
all_sensor_psd = []
all_source_psd = []

freqs_sensor = None
freqs_source = None

# Paths
dynemo_root = paths.dynemo_preprocessing
psd_root = paths.dynemo_plots_PSD_path




# Plot function
def plot_psd_multichannel(freqs, psd_db, title, save_path, mean_label="Mean"):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(psd_db.shape[0]):
        ax.plot(freqs, psd_db[i], alpha=0.20)

    ax.plot(
        freqs,
        psd_db.mean(axis=0),
        color="black",
        linewidth=3,
        label=mean_label,
    )

    ax.axvline(10, linestyle="--", color="gray", linewidth=1)
    ax.set_xlim([fmin, fmax])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# Loop subjects
for subject_code in exp_info.subjects_ids:

    cprint(f"\n>>> Procesando PSD para sujeto {subject_code}")

    # ------------------------------------------------------------
    # SENSOR PSD
    # ------------------------------------------------------------
    cprint(f">>> Cargando sensores procesados para {subject_code}")

    raw_sensor = load.meg(subject_id=subject_code, meg_params={"data_type": "processed"})
    raw_sensor.load_data()
    raw_sensor.pick("mag")

    # Mismo filtro/downsample que el dynemo preprocessing
    raw_sensor.filter(
        l_freq=1,
        h_freq=45,
        method="iir",
        iir_params={"order": 5, "ftype": "butter"},
    )
    raw_sensor.resample(250)

    n_per_seg = int(round(WELCH_WIN_SEC * raw_sensor.info["sfreq"]))

    spectrum_sensor = raw_sensor.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=n_per_seg,
        n_per_seg=n_per_seg,
        n_overlap=0,
        reject_by_annotation=True,
        verbose=False,
    )

    psd_sensor, freqs_sensor = spectrum_sensor.get_data(return_freqs=True)
    sensor_psd_db = 10 * np.log10(psd_sensor)
    # NaN / Inf check - Sensors
    n_nans_sensor = np.isnan(sensor_psd_db).sum()
    n_infs_sensor = np.isinf(sensor_psd_db).sum()

    if n_nans_sensor > 0 or n_infs_sensor > 0:
        rprint(
            f">>> WARNING SENSOR PSD {subject_code}: "
            f"NaNs={n_nans_sensor}, Infs={n_infs_sensor}"
        )
    else:
        gprint(f">>> Sensor PSD OK {subject_code}")

    # Store channel-averaged spectrum (1-D) for the cross-subject grand average.
    all_sensor_psd.append(sensor_psd_db.mean(axis=0))

    sensor_fig_path = os.path.join(psd_root, f"{subject_code}_sensor_psd_by_channel.png")
    os.makedirs(os.path.dirname(sensor_fig_path), exist_ok=True)

    plot_psd_multichannel(
        freqs=freqs_sensor,
        psd_db=sensor_psd_db,
        title=f"Sensor PSD by channel — {subject_code}",
        save_path=sensor_fig_path,
        mean_label="Mean sensors",
    )

    cprint(f">>> PSD sensores guardado en {sensor_fig_path}")

    # ------------------------------------------------------------
    # SOURCE PSD
    # ------------------------------------------------------------
    parc_fif_path = os.path.join(
        dynemo_root,
        subject_code,
        "parcellation",
        f"{subject_code}_lcmv-parc-raw.fif",
    )

    if not os.path.exists(parc_fif_path):
        yprint(f">>> No existe source FIF para {subject_code}: {parc_fif_path}")
        continue

    cprint(f">>> Cargando sources parcelados para {subject_code}")

    raw_source = mne.io.read_raw_fif(parc_fif_path, preload=True)

    n_per_seg_src = int(round(WELCH_WIN_SEC * raw_source.info["sfreq"]))

    spectrum_source = raw_source.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        picks="misc",
        n_fft=n_per_seg_src,
        n_per_seg=n_per_seg_src,
        n_overlap=0,
        reject_by_annotation=True,
        verbose=False,
    )

    psd_source, freqs_source = spectrum_source.get_data(
        picks="misc",
        return_freqs=True,
    )

    source_psd_db = 10 * np.log10(psd_source)
    # NaN / Inf check - Sources
    n_nans_source = np.isnan(source_psd_db).sum()
    n_infs_source = np.isinf(source_psd_db).sum()

    if n_nans_source > 0 or n_infs_source > 0:
        rprint(
            f">>> WARNING SOURCE PSD {subject_code}: "
            f"NaNs={n_nans_source}, Infs={n_infs_source}"
        )
    else:
        gprint(f">>> Source PSD OK {subject_code}")

    # Store parcel-averaged spectrum (1-D) for the cross-subject grand average.
    all_source_psd.append(source_psd_db.mean(axis=0))

    source_fig_path = os.path.join(psd_root,f"{subject_code}_source_psd_by_parcel.png",)
    os.makedirs(os.path.dirname(source_fig_path), exist_ok=True)

    plot_psd_multichannel(
        freqs=freqs_source,
        psd_db=source_psd_db,
        title=f"Source PSD by parcel — {subject_code}",
        save_path=source_fig_path,
        mean_label="Mean parcels",
    )

    cprint(f">>> PSD sources guardado en {source_fig_path}")


# Grand Average
cprint("\n>>> Calculando Grand Average PSD")

if len(all_sensor_psd) > 0:
    # Each entry is a subject's channel-averaged spectrum (1-D). Stack across
    # subjects -> (n_subjects, n_freqs): each line is a subject, black = mean.
    grand_sensor_psd = np.stack(all_sensor_psd, axis=0)

    ga_sensor_path = os.path.join( psd_root, "grand_average_psd_sensors.png",)

    plot_psd_multichannel(
        freqs=freqs_sensor,
        psd_db=grand_sensor_psd,
        title=f"Grand Average PSD — Sensors (N={len(all_sensor_psd)})",
        save_path=ga_sensor_path,
        mean_label="Mean across subjects",
    )

    cprint(f">>> Grand Average sensores guardado en {ga_sensor_path}")

else:
    yprint(">>> No se calculó Grand Average de sensores: lista vacía.")


if len(all_source_psd) > 0:
    # Each entry is a subject's parcel-averaged spectrum (1-D). Stack across
    # subjects -> (n_subjects, n_freqs): each line is a subject, black = mean.
    grand_source_psd = np.stack(all_source_psd, axis=0)

    ga_source_path = os.path.join(
        psd_root,
        "grand_average_psd_sources.png",
    )

    plot_psd_multichannel(
        freqs=freqs_source,
        psd_db=grand_source_psd,
        title=f"Grand Average PSD — Sources (N={len(all_source_psd)})",
        save_path=ga_source_path,
        mean_label="Mean across subjects",
    )

    cprint(f">>> Grand Average sources guardado en {ga_source_path}")

else:
    yprint(">>> No se calculó Grand Average de sources: lista vacía.")


cprint("\n>>> PSD DYNEMO PREPROCESSING TERMINADO")