"""Run DyNeMo modules I -> VII one after another (params come from dynemo_config)."""
import os
import subprocess
import sys


try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = r"D:\OneDrive - The University of Nottingham\OPM-MEG-analysis - OPM2\Scripts\dynemo"

sys.path.insert(0, HERE)                   # dynemo_config
sys.path.insert(0, os.path.dirname(HERE))  # paths

import paths
import setup
from dynemo_config import ch_picks


def preprocessing_complete(preprocessing_path, subjects):
    """Whether every subject has both artifacts stage I writes last."""
    return bool(subjects) and all(
        os.path.exists(os.path.join(preprocessing_path, subject, "parcellation",
                                    f"{subject}_parcel_data_spatial_basis_symmetric.npy"))
        and os.path.exists(os.path.join(preprocessing_path, subject,
                                        f"{subject}_dynemo_preprocessing_summary.txt"))
        for subject in subjects
    )


# Preprocessing is expensive: only run it if some subject is still missing output.
# The folder alone is not enough, it appears as soon as the first subject starts.
preproc_path = paths.dynemo_preprocessing_path(ch_picks)
if preprocessing_complete(preproc_path, setup.exp_info().subjects_ids):
    print(f"[run_all] Skipping preprocessing, found {preproc_path}")
else:
    subprocess.run([sys.executable, os.path.join(HERE, "dynemo_I_preprocessing.py")], check=True)

subprocess.run([sys.executable, os.path.join(HERE, "dynemo_II_model.py")], check=True)
subprocess.run([sys.executable, os.path.join(HERE, "dynemo_III_regression_spectra.py")], check=True)
subprocess.run([sys.executable, os.path.join(HERE, "dynemo_IV_plotting_networks.py")], check=True)
subprocess.run([sys.executable, os.path.join(HERE, "dynemo_V_mixing_coefficients.py")], check=True)
subprocess.run([sys.executable, os.path.join(HERE, "dynemo_VI_temporal_analysis.py")], check=True)
subprocess.run([sys.executable, os.path.join(HERE, "dynemo_VII_trf_mixing_coefficients.py")], check=True)


