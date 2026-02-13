from osl_dynamics.models import load
import paths
import os
from osl_dynamics.data import Data
import pickle
import numpy as np
from osl_dynamics.utils import plotting
from osl_dynamics.inference import modes
from scipy import stats
from osl_dynamics.inference import modes
import matplotlib.pyplot as plt
import zipfile



# Paths:
dynemo_infered_parameters_path = os.path.join(paths.dynemo_infered_results_path)
dynemo_plots_mixing_coefficients_path = os.path.join(paths.plots_path, "DyNeMo/Mixing Coefficients")
os.makedirs(dynemo_plots_mixing_coefficients_path, exist_ok=True)
dynemo_generic_data = paths.dynemo_generic_data_path
dynemo_prepared_data_path_PARA_PLOTS = os.path.join(dynemo_generic_data, "prepared_data_for_plots")




#-----------------------------LOADING DATA-----------------------------

print("\033[1;36m \n\n\n ---------------------------------------------- \033[0m")
print("\033[1;36m       Loading Infered Parameters \033[0m")
print("\033[1;36m ---------------------------------------------- \n \033[0m")

# alpha = pickle.load(open(os.path.join(dynemo_infered_parameters_path, "alp.pkl"), "rb"))
# print(f"\033[1;36m   >>>     Parámetros ALPHA cargados desde: {dynemo_infered_parameters_path}\033[0m")
# means = np.load(os.path.join(dynemo_infered_parameters_path, "means.npy"))
# covs = np.load(os.path.join(dynemo_infered_parameters_path, "covs.npy"))
# print(f"\033[1;36m   >>>     Parámetros MEANS y COVS cargados desde: {dynemo_infered_parameters_path}\033[0m")


#-----BORRAR DESDE ACA------

def get_inf_params(name, rename):
    """Descarga parámetros inferidos pre-calculados desde OSF"""
    zip_filename = f"{name}.zip"
    
    print(f"\033[1;36m   >>>     Descargando {name}.zip desde OSF/inf_params...\033[0m")
    result = os.system(f"osf -p by2tc fetch inf_params/{name}.zip")
    # Crear directorio destino
    os.makedirs(rename, exist_ok=True)
    
    # Extraer con zipfile 
    print(f"\033[1;36m   >>>     Extrayendo archivos...\033[0m")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(rename)
    os.remove(zip_filename)
    
    print(f"\033[1;32m   ✓ Parámetros inferidos descargados en: {rename}\033[0m")
    return f"Data downloaded to: {rename}"

# Descargar parámetros inferidos (aproximadamente 11 MB)
# Ajusta la ruta según tus necesidades
inf_params_path = os.path.join(dynemo_prepared_data_path_PARA_PLOTS, "inf_params")
get_inf_params("tde_dynemo_notts_mrc_meguk_giles_5_subjects", 
               rename=inf_params_path)

alpha = pickle.load(open(os.path.join(inf_params_path, "alp.pkl"), "rb"))

# --- BORRAR HASTA aca ---


#-----------------------------PLOTTING-----------------------------

print("\033[1;36m \n\n\n ---------------------------------------------- \033[0m")
print("\033[1;36m       Plotting Mixing Coefficients \033[0m")
print("\033[1;36m ---------------------------------------------- \n \033[0m")


# Using the utils.plotting.plot_alpha to plot the mixing coefficients.
# Plot the mixing coefficient time course for the first subject (8 seconds)
print(f"\033[1;36m   >>>     Plotteando mixing coefficients raw\n \033[0m")
fig, ax = plotting.plot_alpha(alpha[0], n_samples=2000)
fig.savefig(os.path.join(dynemo_plots_mixing_coefficients_path, "mixing_coefficients_raw.png"))
print(f"\033[1;36m   >>>     Plot guardado correctamente en {dynemo_plots_mixing_coefficients_path}\n \033[0m")



print(f"\033[1;36m   >>>     Plotteando mixing coefficients normalized\n \033[0m")
# Normalizing the mixing coefficients
covs = np.load(os.path.join(inf_params_path, "covs.npy"))
norm_alpha = modes.reweight_alphas(alpha, covs)
fig, ax = plotting.plot_alpha(norm_alpha[0], n_samples=2000)
fig.savefig(os.path.join(dynemo_plots_mixing_coefficients_path, "mixing_coefficients_normalized.png"))
print(f"\033[1;36m   >>>     Plot guardado correctamente en {dynemo_plots_mixing_coefficients_path}\n \033[0m")



# Summary Statistics: time average mixing coefficient.
print(f"\033[1;36m   >>>     Plotteando summary statistics: time average mixing coefficient\n \033[0m")
mean_norm_alpha = np.array([np.mean(a, axis=0) for a in norm_alpha])
# Print group average
print(np.mean(mean_norm_alpha, axis=0))
# Plot distribution over subjects
fig, ax = plotting.plot_violin(mean_norm_alpha.T, x_label="Mode", y_label="Mean alpha")
fig.savefig(os.path.join(dynemo_plots_mixing_coefficients_path, "summary_statistics.png"))
print(f"\033[1;36m   >>>     Plot guardado correctamente en {dynemo_plots_mixing_coefficients_path}\n \033[0m")



# Summary Statistics: standard deviation of the mixing coefficient time course
print(f"\033[1;36m   >>>     Plotteando summary statistics: std of mixing coefficient time course\n \033[0m")
std_norm_alpha = np.array([np.std(a, axis=0) for a in norm_alpha])
# Print group average
print(np.mean(std_norm_alpha, axis=0))
# Plot distribution over subjects
fig, ax = plotting.plot_violin(std_norm_alpha.T, x_label="Mode", y_label="Std alpha")
fig.savefig(os.path.join(dynemo_plots_mixing_coefficients_path, "summary_statistics_std.png"))
print(f"\033[1;36m   >>>     Plot guardado correctamente en {dynemo_plots_mixing_coefficients_path}\n \033[0m")



# Summary Statistics: kurtosis, which reflects spiking the in mixing coefficient time course.
print(f"\033[1;36m   >>>     Plotteando summary statistics: kurtosis of mixing coefficient time course\n \033[0m")
kurt_norm_alpha = np.array([stats.kurtosis(a, axis=0) for a in norm_alpha])
# Print group average
print(np.mean(kurt_norm_alpha, axis=0))
# Plot distribution over subjects
fig, ax = plotting.plot_violin(kurt_norm_alpha.T, x_label="Mode", y_label="Kurt. alpha")
fig.savefig(os.path.join(dynemo_plots_mixing_coefficients_path, "summary_statistics_kurtosis.png"))
print(f"\033[1;36m   >>>     Plot guardado correctamente en {dynemo_plots_mixing_coefficients_path}\n \033[0m")


# Summary Statistics: Binarized mixing coefficients (argmax)
print(f"\033[1;36m   >>>     Plotteando summary statistics: binarized mixing coefficient time course\n \033[0m")
# Use the mode with the largest value to define mode activations
atc = modes.argmax_time_courses(norm_alpha)
# Plot the mode activation time course for the first subject (first 8 seconds)
fig, ax = plotting.plot_alpha(atc[0], n_samples=2000)
fig.savefig(os.path.join(dynemo_plots_mixing_coefficients_path, "summary_statistics_binarized_argmax_time_course.png"))
print(f"\033[1;36m   >>>     Plot guardado correctamente en {dynemo_plots_mixing_coefficients_path}\n \033[0m")



# Summary Statistics: Binarize with a percentile threshold
print(f"\033[1;36m   >>>     Plotteando summary statistics: binarized (percentile) mixing coefficient time course\n \033[0m")
def plot_hist(alpha, x_label=None):
    fig, ax = plt.subplots()
    for i in range(alpha.shape[1]):
        ax.hist(alpha[:, i], label=f"Mode {i+1}", bins=50, histtype="step")
    ax.legend()
    ax.set_xlabel(x_label)
    return fig 

# Concatenate the alphas for each subject
concat_norm_alpha = np.concatenate(norm_alpha)

# Plot their distribution
fig = plot_hist(concat_norm_alpha, x_label="Normalized mixing coefficients")  # ← CAPTURA fig
fig.savefig(os.path.join(dynemo_plots_mixing_coefficients_path, "summary_statistics_percentile_histogram.png"))
print(f"\033[1;36m   >>>     Plot guardado correctamente en {dynemo_plots_mixing_coefficients_path}\n \033[0m")


# Let’s use the 90th percentile to determine the threshold of each mode.
thres = np.array([np.percentile(a, 90, axis=0) for a in norm_alpha])
# Binarize the mixing coefficients
thres_norm_alpha = []
n_subjects = len(norm_alpha)
for na, t in zip(norm_alpha, thres):
    tna = (na > t).astype(int)
    thres_norm_alpha.append(tna)

# Plot the mode activation time courses for the first subject (first 8 seconds)
fig, ax = plotting.plot_separate_time_series(thres_norm_alpha[0], n_samples=2000)
fig.savefig(os.path.join(dynemo_plots_mixing_coefficients_path, "summary_statistics_binarized_percentile_time_course.png"))
print(f"\033[1;36m   >>>     Plot guardado correctamente en {dynemo_plots_mixing_coefficients_path}\n \033[0m")



# Summary Statistics: Binarize with a GMM threshold
print(f"\033[1;36m   >>>     Plotteando summary statistics: binarized (GMM) mixing coefficient time course\n \033[0m")
thres_norm_alpha = modes.gmm_time_courses(norm_alpha)
# Plot the mode activation time courses for the first subject (first 8 seconds)
fig, ax = plotting.plot_separate_time_series(thres_norm_alpha[0], n_samples=2000)
fig.savefig(os.path.join(dynemo_plots_mixing_coefficients_path, "summary_statistics_binarized_gmm_time_course.png"))
print(f"\033[1;36m   >>>     Plot guardado correctamente en {dynemo_plots_mixing_coefficients_path}\n \033[0m")
# Display the group average
fo = modes.fractional_occupancies(thres_norm_alpha)
print(np.mean(fo, axis=0), 2)
