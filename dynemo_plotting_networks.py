import paths
import os
import numpy as np
from osl_dynamics.utils import plotting
from osl_dynamics.analysis import power
from osl_dynamics.analysis import connectivity
import zipfile




# Paths:
dynemo_generic_data = paths.dynemo_generic_data_path
dynemo_prepared_data_path = os.path.join(dynemo_generic_data, "prepared_data")
dynemo_prepared_data_path_PARA_PLOTS = os.path.join(dynemo_generic_data, "prepared_data_for_plots")
dynemo_object_data_path = os.path.join(dynemo_generic_data, "data_object")
dynemo_trained_data_path = os.path.join(dynemo_generic_data, "trained_model")
dynemo_infered_parameters_path = os.path.join(paths.dynemo_infered_results_path)
data_object_file = os.path.join(dynemo_object_data_path, "data.pkl")
spectra_data_path = paths.dynemo_spectra_path
dynemo_plots_PSD_path = os.path.join(paths.plots_path, "DyNeMo/PSD")
os.makedirs(dynemo_plots_PSD_path, exist_ok=True)
dynemo_plots_power_map_path = os.path.join(paths.plots_path, "DyNeMo/Power Maps")
os.makedirs(dynemo_plots_power_map_path, exist_ok=True)
dynemo_plots_coherence_networks_path = os.path.join(paths.plots_path, "DyNeMo/Coherence Networks")
os.makedirs(dynemo_plots_coherence_networks_path, exist_ok=True)
dynemo_plots_coherence_maps_path = os.path.join(paths.plots_path, "DyNeMo/Coherence Maps")
os.makedirs(dynemo_plots_coherence_maps_path, exist_ok=True)



#-----------------------------LOAD SPECTRA -----------------------------

print("\033[1;36m \n\n\n ---------------------------------------------- \033[0m")
print("\033[1;36m               Loading Data \033[0m")
print("\033[1;36m ---------------------------------------------- \n \033[0m")

# # Esto REEMPLAZARLO POR MIS DATOS GUARDADOS DE SPECTRA
# print(f"\033[1;36m Cargando datos desde: {spectra_data_path} \033[0m")
# f = np.load(os.path.join(spectra_data_path, "f.npy"))
# psd = np.load(os.path.join(spectra_data_path, "psd.npy"))
# coh = np.load(os.path.join(spectra_data_path, "coh.npy"))
# w = np.load(os.path.join(spectra_data_path, "w.npy"))

# Esto ELIMINARLO DESPUES-------------------
# Download the dataset from OSL using osfclient

def get_spectra(name, rename):
    zip_filename = f"{name}.zip"
    print(f"\033[1;36m   >>>     Descargando {name}.zip desde OSF/spectra...\033[0m")
    result = os.system(f"osf -p by2tc fetch spectra/{name}.zip")

    # Crear directorio destino
    os.makedirs(rename, exist_ok=True)
    
    # Extraer con zipfile
    print(f"\033[1;36m   >>>     Extrayendo archivos...\033[0m")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(rename)
    os.remove(zip_filename)
    print(f"\033[1;36m   ✓ Espectros descargados en: {rename}\033[0m")
    return f"Data downloaded to: {rename}"

# Descargar espectros (aproximadamente 7 MB)
get_spectra("tde_dynemo_notts_mrc_meguk_giles_5_subjects", 
            rename=dynemo_prepared_data_path_PARA_PLOTS)
#------------BORRAR HASTA ACÁ-------------------


f = np.load(os.path.join(dynemo_prepared_data_path_PARA_PLOTS, "f.npy"))
psd = np.load(os.path.join(dynemo_prepared_data_path_PARA_PLOTS, "psd.npy"))
coh = np.load(os.path.join(dynemo_prepared_data_path_PARA_PLOTS, "coh.npy"))
w = np.load(os.path.join(dynemo_prepared_data_path_PARA_PLOTS, "w.npy"))
# Esto ELIMINARLO DESPUES-------------------


# Imprimir resumen
print(f"\033[1;36m   >>>     Datos cargados: \033[0m")
print(f"\033[1;36m   Shape Frequencies: {f.shape}\033[0m")
print(f"\033[1;36m\n   Shape PSD: {psd.shape}\033[0m")
print(f"\033[1;36m\n   Shape Coherence: {coh.shape}\033[0m")
print(f"\033[1;36m\n   Shape Weights: {w.shape}\033[0m")



#-----------------------------PLOTTING -----------------------------

print("\033[1;36m \n\n\n ---------------------------------------------- \033[0m")
print("\033[1;36m                Plotting \033[0m")
print("\033[1;36m ---------------------------------------------- \n \033[0m")


# 1. Plotear PSD promedio grupal para cada modo.
psd_coefs = psd[:, 0]
print(psd_coefs.shape)
print("\033[1;36m   >>>     Ploteando promedio grupal PSD para cada modo... \n \033[0m")
# Average over subjects and channels
psd_coefs_mean = np.mean(psd_coefs, axis=(0,2))
print(psd_coefs_mean.shape)

# Plot
n_modes = psd_coefs_mean.shape[0]
fig, ax = plotting.plot_line(
    [f] * n_modes,
    psd_coefs_mean,
    labels=[f"Mode {i}" for i in range(1, n_modes + 1)],
    x_label="Frequency (Hz)",
    y_label="PSD (a.u.)",
    x_range=[f[0], f[-1]],
)

print("\033[1;36m   >>>     Guardando plot de PSD grupal... \n \033[0m")
fig.savefig(os.path.join(dynemo_plots_PSD_path, "group_mean_PSD.png"))
print(f"\033[1;36m   >>>     Plot de PSD grupal guardado correctamente en {os.path.join(dynemo_plots_PSD_path, 'group_mean_PSD.png')}\n \033[0m")



# 2. Plotear Power Maps
print("\033[1;36m   >>>     Ploteando y guardando power map... \n \033[0m")
p = power.variance_from_spectra(f, psd_coefs)
mean_p = np.average(p, axis=0, weights=w)
power.save(
    mean_p,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    subtract_mean=True,
    filename=os.path.join(dynemo_plots_power_map_path, "power_map_promedio.png") 
)
print(f"\033[1;36m   >>>     Plot de power map guardado correctamente en {os.path.join(dynemo_plots_power_map_path, 'power_map_promedio.png')}\n \033[0m")



# 3. Plotear Coherence networks
print("\033[1;36m   >>>     Ploteando y guardando coherence networks... \n \033[0m")
c = connectivity.mean_coherence_from_spectra(f, coh)
# Promedio en sujetos
mean_c = np.average(c, axis=0, weights=w)

# Threshold del top 3% relativo al promedio
thres_mean_c = connectivity.threshold(mean_c, percentile=97, subtract_mean=True)

connectivity.save(
    thres_mean_c,
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    plot_kwargs={"edge_vmin": 0, "edge_vmax": np.max(thres_mean_c), "edge_cmap": "Reds"},
    filename=os.path.join(dynemo_plots_coherence_networks_path, "coherence_networks_promedio.png")
)
print(f"\033[1;36m   >>>     Plot de coherence networks guardado correctamente en {os.path.join(dynemo_plots_coherence_networks_path, 'coherence_networks_promedio.png')}\n \033[0m")



# 4. Plotear Coherence maps
print("\033[1;36m   >>>     Ploteando y guardando coherence maps... \n \033[0m")
mean_c_map = connectivity.mean_connections(mean_c)
power.save(
    mean_c_map,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    subtract_mean=True,
    filename=os.path.join(dynemo_plots_coherence_maps_path, "coherence_maps_promedio.png"),
)
print(f"\033[1;36m   >>>     Plot de coherence maps guardado correctamente en {os.path.join(dynemo_plots_coherence_maps_path, 'coherence_maps_promedio.png')}\n \033[0m")


print(f"\033[1;36m   >>>     Todos los plots se guardaron correctamente!!\n \033[0m")
