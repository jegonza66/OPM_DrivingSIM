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

from osl_dynamics.models import load
import paths
from osl_dynamics.data import Data
import pickle
import numpy as np
from osl_dynamics.inference import modes
from osl_dynamics.analysis import spectral
import setup
from general_utility_functions import cprint, rprint, yprint, gprint

# Setup
from dynemo_config import n_modes, n_embeddings, sequence_length

# Paths:
dynemo_object_data_path = paths.dynemo_run_save_path(n_modes, n_embeddings, sequence_length, "DyNeMo_Object_Data")
dynemo_trained_data_path = paths.dynemo_run_save_path(n_modes, n_embeddings, sequence_length, "DyNeMo_Trained_Model")
data_object_file =  os.path.join(dynemo_object_data_path, "data.pkl")
dynemo_infered_parameters_path = paths.dynemo_run_save_path(n_modes, n_embeddings, sequence_length, "DyNeMo_Infered_Parameters")
spectra_data_path = paths.dynemo_run_save_path(n_modes, n_embeddings, sequence_length, "DyNeMo_Spectra")
raw_data_file = os.path.join(dynemo_object_data_path, "raw_data.pkl")
os.makedirs(dynemo_infered_parameters_path, exist_ok=True)
os.makedirs(spectra_data_path, exist_ok=True)

exp_info = setup.exp_info()


#-----------------------------LOADING DATA-----------------------------#

cprint(" \n\n\n ----------------------------------------------  ")
cprint("       Loading Trained Model and Data  ")
cprint(" ---------------------------------------------- \n  ")

# Loading prepared raw data and TDE-PCA data
data = Data(raw_data_file)
prepared_data = Data(data_object_file)

cprint(f"   >>>     Objeto Data cargado desde: {data_object_file} ")
cprint(f"   >>>     Número de sujetos/sesiones: {data.n_sessions} ")
cprint(f"   >>>     Número de canales: {data.n_channels} ")
cprint(f"   >>>     Orden de dimensiones: (samples, channels) ")

# Loading trained model
model = load(dynemo_trained_data_path)
cprint(f"   >>>     Modelo entrenado cargado desde: {dynemo_trained_data_path} ")


#-----------------------------INFERED PARAMETERS-----------------------------#

cprint(" \n\n\n ----------------------------------------------  ")
cprint("       Getting Infered Parameters  ")
cprint(" ---------------------------------------------- \n  ")

# Trim data
trimmed_data = data.trim_time_series(n_embeddings=n_embeddings, sequence_length=sequence_length)
alpha = model.get_alpha(prepared_data)

# Checking alphas
for a, x in zip(alpha, trimmed_data):
    print(a.shape, x.shape)
    if a.shape[0] != x.shape[0]:
         rprint("   >>>     Se usó un valor incorrecto para n_embeddings o sequence_length al recortar los datos. ")

# Info
cprint(f"   >>>     Alphas obtenidos para: {len(alpha)} sujetos ")
cprint(f"   >>>     Modos detectados:      {alpha[0].shape[1]} (Debería ser igual a n_modes del config) ")
cprint(f"   >>>     Shape (Ej. Sujeto 1):  {alpha[0].shape} (tiempo, modos) ")

# Save alphas
cprint(f"   >>>     Guardando los alpha en {dynemo_infered_parameters_path}  ")
os.makedirs(dynemo_infered_parameters_path, exist_ok=True)
pickle.dump(alpha, open(os.path.join(dynemo_infered_parameters_path, "alp.pkl"), "wb"))

# Get inferred state/mode means and covariances
cprint("   >>>     Obteniendo means y covariances ")
means, covs = model.get_means_covariances()

# Save alphas means and covariances
cprint(f"   >>>     Guardando means y covariances en {dynemo_infered_parameters_path}  ")
np.save(os.path.join(dynemo_infered_parameters_path, "means.npy"), means)
np.save(os.path.join(dynemo_infered_parameters_path, "covs.npy"), covs)

# Reweight the mixing coefficients.
alpha = modes.reweight_alphas(alpha, covs)
pickle.dump(alpha, open(os.path.join(dynemo_infered_parameters_path, "alp_reweighted.pkl"), "wb"))

# Calculate regression spectra for each mode and subject (will take a few minutes)
f, psd, coh, w = spectral.regression_spectra(
    data=trimmed_data,
    alpha=alpha,
    sampling_frequency=250,
    frequency_range=[1, 45],
    window_length=1000,
    step_size=20,
    n_sub_windows=8,
    return_coef_int=True,
    return_weights=True,
)

# Check results
cprint(f"f.shape: {f.shape}, → debería ser 1D (p.ej. (45,))")
cprint(f"psd.shape: {psd.shape}, → debería ser (subjects, 2, states, channels, frequencies)")
cprint(f"coh.shape: {coh.shape}, → debería ser (subjects, states, channels, channels, frequencies)")
cprint(f"w.shape: {w.shape}, → debería ser (subjects,)")


np.save(os.path.join(spectra_data_path, "f.npy"), f)
cprint(" f guardado en spectra_data_path/f.npy  ")
np.save(os.path.join(spectra_data_path, "psd.npy"), psd)
cprint(" psd guardado en spectra_data_path/psd.npy  ")
np.save(os.path.join(spectra_data_path, "coh.npy"), coh)
cprint(" coh guardado en spectra_data_path/coh.npy  ")
np.save(os.path.join(spectra_data_path, "w.npy"), w)
cprint(" w guardado en spectra_data_path/w.npy  ")