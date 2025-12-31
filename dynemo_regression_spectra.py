from osl_dynamics.models import load
import paths
import os
from osl_dynamics.data import Data
import pickle
import numpy as np
from osl_dynamics.inference import modes
from osl_dynamics.analysis import spectral


# Paths:
dynemo_generic_data = paths.dynemo_generic_data_path
dynemo_prepared_data_path = os.path.join(dynemo_generic_data, "prepared_data")
dynemo_object_data_path = os.path.join(dynemo_generic_data, "data_object")
dynemo_trained_data_path = os.path.join(dynemo_generic_data, "trained_model")
dynemo_infered_parameters_path = os.path.join(paths.dynemo_infered_results_path)
data_object_file = os.path.join(dynemo_object_data_path, "data.pkl")
spectra_data_path = paths.dynemo_spectra_path



#-----------------------------LOADING DATA-----------------------------

print("\033[1;36m \n\n\n ---------------------------------------------- \033[0m")
print("\033[1;36m       Loading Data \033[0m")
print("\033[1;36m ---------------------------------------------- \n \033[0m")


# Loading prepared data
data_object = Data(dynemo_prepared_data_path)
print(f"\033[1;36m   >>>     Objeto Data cargado desde: {data_object_file}\033[0m")
print(f"\033[1;36m\n   ===== INFORMACIÓN DEL DATASET =====\033[0m")
print(f"\033[1;36m   >>>     Número de sujetos/sesiones: {data_object.n_sessions}\033[0m")
print(f"\033[1;36m   >>>     Número de canales: {data_object.n_channels}\033[0m")
print(f"\033[1;36m   >>>     Total de muestras (todos los sujetos): {data_object.n_samples}\033[0m")

print(f"\033[1;36m\n   ===== FORMATO =====\033[0m")
print(f"\033[1;36m   >>>     Orden de dimensiones: (samples, channels) ✓\033[0m")
print(f"\033[1;36m   >>>     Primera dimensión = tiempo/muestras\033[0m")
print(f"\033[1;36m   >>>     Segunda dimensión = canales/sensores\033[0m")

# Loading trained model
model = load(dynemo_trained_data_path)
print(f"\033[1;36m   >>>     Modelo entrenado cargado desde: {dynemo_trained_data_path}\033[0m")




#-----------------------------INFERED PARAMETERS-----------------------------

alpha = pickle.load(open(os.path.join(dynemo_infered_parameters_path, "alp.pkl"), "rb"))
print(f"\033[1;36m   >>>     Parámetros ALPHA cargados desde: {dynemo_infered_parameters_path}\033[0m")
means = np.load(os.path.join(dynemo_infered_parameters_path, "means.npy"))
covs = np.load(os.path.join(dynemo_infered_parameters_path, "covs.npy"))
print(f"\033[1;36m   >>>     Parámetros MEANS y COVS cargados desde: {dynemo_infered_parameters_path}\033[0m")

# When we separate the data into sequences we lose time points from the end of the time series, we need to trim the source-space data to match the inferred state probabilities (alpha).
trimmed_data = data_object.trim_time_series(sequence_length=100)

# Before calculate the mode spectra, we reweight the mixing coefficients using the trace of each mode covariance
alpha_reweighted = modes.reweight_alphas(alpha, covs)

# Save the updated infered parameters
print(f"\033[1;36m   >>>     Guardando los alpha reweighteados en {dynemo_infered_parameters_path} \033[0m")
pickle.dump(alpha_reweighted, open(os.path.join(dynemo_infered_parameters_path, "alp.pkl"), "wb"))



#-----------------------------POWER SPECTRA -----------------------------

print("\033[1;36m \n\n\n ---------------------------------------------- \033[0m")
print("\033[1;36m       Calculating Spectra \033[0m")
print("\033[1;36m ---------------------------------------------- \n \033[0m")


# Calculate regression spectra for each mode and subject
print(f"\033[1;36m   >>>     Calculando espectros de regresión \033[0m")
print(f"\033[1;36m   >>>     Podría tomar unos minutos \033[0m")
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

print("\033[1;36m\n   ===== REGRESSION SPECTRA RESULTS =====\033[0m")

print(f"\033[1;36m   Shape Frequencies: {f.shape}\033[0m")
print(f"\033[1;36m   Range: [{f[0]:.2f}, {f[-1]:.2f}] Hz\033[0m")
print(f"\033[1;36m\n   Shape PSD: {psd.shape}\033[0m")
print(f"\033[1;36m   Format PSD: (n_subjects={psd.shape[0]}, n_modes={psd.shape[1]}, n_freqs={psd.shape[2]}, n_channels={psd.shape[3]})\033[0m")
print(f"\033[1;36m\n   Shape Coherence: {coh.shape}\033[0m")
print(f"\033[1;36m   Format Coherence: (n_subjects={coh.shape[0]}, n_modes={coh.shape[1]}, n_freqs={coh.shape[2]}, n_channels={coh.shape[3]}, n_channels={coh.shape[4]})\033[0m")
print(f"\033[1;36m\n   Shape Weights: {w.shape}\033[0m")

print("\033[1;32m\n   ✓ Regression spectra computed successfully\033[0m")


print(f"\033[1;36m   >>>     Guardando espectros... \033[0m")

# Guardar cada array como archivo .npy
np.save(os.path.join(spectra_data_path, "f.npy"), f)
np.save(os.path.join(spectra_data_path, "psd.npy"), psd)
np.save(os.path.join(spectra_data_path, "coh.npy"), coh)
np.save(os.path.join(spectra_data_path, "w.npy"), w)

print(f"\033[1;36m   >>>     Espectros guardados correctamente en: {spectra_data_path} \033[0m")
