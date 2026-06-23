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

import paths
import numpy as np
import setup
from general_utility_functions import cprint, rprint, yprint, gprint
from osl_dynamics.utils import plotting
import pickle

# Setup
exp_info = setup.exp_info()

# Run parameters (must match the trained model in dynemo_II / dynemo_III)
n_modes = 6
n_embeddings = 15
sequence_length = 100

# Paths:
spectra_data_path = paths.dynemo_run_save_path(n_modes, n_embeddings, sequence_length, "DyNeMo_Spectra")
dynemo_infered_parameters_path = paths.dynemo_run_save_path(n_modes, n_embeddings, sequence_length, "DyNeMo_Infered_Parameters")
dynemo_plots_mixing_coefficients_path = paths.dynemo_run_plots_path(n_modes, n_embeddings, sequence_length, "Mixing_Coefficients")
os.makedirs(dynemo_plots_mixing_coefficients_path, exist_ok=True)
ALP_PATH = os.path.join(dynemo_infered_parameters_path, "alp.pkl")
ALP_PLOT_PATH = os.path.join(dynemo_plots_mixing_coefficients_path, "mixing_coefficients.png")
ALP_REWEIGHTED_PATH = os.path.join(dynemo_infered_parameters_path, "alp_reweighted.pkl")
ALP_REWEIGHTED_PLOT_PATH = os.path.join(dynemo_plots_mixing_coefficients_path, "mixing_coefficients_weighted.png")


####### ALPHA #######
cprint(">>> Cargando alpha...")
with open(ALP_PATH, "rb") as f:
    alpha = pickle.load(f)

# Plot the mixing coefficient time course for the first subject (8 seconds)
fig, ax = plotting.plot_alpha(alpha[0], n_samples=2000)

cprint(">>> Guardando plot de alpha en...")
fig.savefig(ALP_PLOT_PATH, dpi=300)


####### ALPHA REWEIGHTED #######
cprint(">>> Cargando alpha reweighted...")
with open(ALP_REWEIGHTED_PATH, "rb") as f:
    alpha = pickle.load(f)

# Plot the reweighted mixing coefficient time course for the first subject (8 seconds)
fig, ax = plotting.plot_alpha(alpha[0], n_samples=2000)

cprint(">>> Guardando plot de alpha reweighted en...")
fig.savefig(ALP_REWEIGHTED_PLOT_PATH, dpi=300)