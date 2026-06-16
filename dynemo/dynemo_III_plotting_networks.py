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
from osl_dynamics.utils import plotting
from osl_dynamics.analysis import power
from osl_dynamics.analysis import connectivity
import matplotlib.pyplot as plt
import setup
from general_utility_functions import cprint, rprint, yprint, gprint
from osl_dynamics import files


# Setup
exp_info = setup.exp_info()

# Paths:
dynemo_prepared_data_path = paths.dynemo_prepared_data_path
dynemo_object_data_path = paths.dynemo_object_data_path
dynemo_trained_data_path = paths.dynemo_trained_data_path
data_object_file =  os.path.join(dynemo_object_data_path, "data.pkl")
spectra_data_path = paths.dynemo_spectra_path
dynemo_plots_PSD_path = paths.dynemo_plots_PSD_path
dynemo_plots_power_map_path = paths.dynemo_plots_power_map_path
dynemo_plots_coherence_networks_path = paths.dynemo_plots_coherence_networks_path
dynemo_plots_coherence_maps_path = paths.dynemo_plots_coherence_maps_path
subjects_dir = os.path.join(paths.mri_path, 'freesurfer')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Ensure plot output directories exist
for _plot_dir in [dynemo_plots_PSD_path, dynemo_plots_power_map_path,
                  dynemo_plots_coherence_networks_path, dynemo_plots_coherence_maps_path]:
    os.makedirs(_plot_dir, exist_ok=True)


#-----------------------------LOAD SPECTRA -----------------------------

cprint(" \n\n\n ----------------------------------------------  ")
cprint("               Loading Data  ")
cprint(" ---------------------------------------------- \n  ")

# Load spectra
cprint(f" Cargando datos desde: {spectra_data_path}  ")
f = np.load(os.path.join(spectra_data_path, "f.npy"))
psd = np.load(os.path.join(spectra_data_path, "psd.npy"))
coh = np.load(os.path.join(spectra_data_path, "coh.npy"))
w = np.load(os.path.join(spectra_data_path, "w.npy"))

# Check data
cprint(f"f.shape: {f.shape}, → debería ser 1D (p.ej. (45,))")
cprint(f"psd.shape: {psd.shape}, → debería ser (subjects, 2, states, channels, frequencies)")
cprint(f"coh.shape: {coh.shape}, → debería ser (subjects, states, channels, channels, frequencies)")
cprint(f"w.shape: {w.shape}, → debería ser (subjects,)")



#-----------------------------PLOTTING -----------------------------

cprint(" \n\n\n ----------------------------------------------  ")
cprint("                Plotting  ")
cprint(" ---------------------------------------------- \n  ")
cprint("   >>>     Ploteando PSD de coeficientes por modo...")

# psd shape:
# subjects x 2 x modes x channels x frequencies

psd_coefs = psd[:, 0]

# promedio sobre sujetos y canales
psd_coefs_mean = np.mean(psd_coefs, axis=(0,2))

n_modes = psd_coefs_mean.shape[0]

# colores fijos para cada modo
mode_colors = [
    "tab:blue",     # mode 1
    "tab:red",      # mode 2
    "tab:green",    # mode 3
    "tab:orange",   # mode 4
    "tab:purple",   # mode 5
    "tab:brown",    # mode 6
    "tab:pink",     # mode 7
    "tab:gray",     # mode 8
]

############# PSD TODOS LOS MODOS #############

fig, ax = plt.subplots(figsize=(8,5))

for i in range(n_modes):

    ax.plot(
        f,
        psd_coefs_mean[i],
        label=f"Mode {i+1}",
        color=mode_colors[i],
        linewidth=2
    )

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD coefficient (a.u.)")

ax.set_xlim([f[0], f[-1]])

ax.legend()

fig.savefig(
    os.path.join(
        dynemo_plots_PSD_path,
        "group_mean_PSD_coefficients.png"
    )
)

cprint(
    f"   >>>     PSD coeficientes guardado en "
    f"{os.path.join(dynemo_plots_PSD_path, 'group_mean_PSD_coefficients.png')}"
)

############# PSD POR MODO SEPARADO #############

for i in range(n_modes):

    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(
        f,
        psd_coefs_mean[i],
        color=mode_colors[i],
        linewidth=2,
        label=f"Mode {i+1}"
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD coefficient (a.u.)")

    ax.set_xlim([f[0], f[-1]])

    # Cada modo se autoescala en y para que su PSD sea visible
    ax.legend()

    out_file = os.path.join(
        dynemo_plots_PSD_path,
        f"group_mean_PSD_coefficients_mode_{i+1}.png"
    )

    fig.savefig(out_file)

    cprint(
        f"   >>>     PSD coeficientes Mode {i+1} guardado en {out_file}"
    )


############### PSD without dominant mode ###############

exclude_modes = [1]  # mode 2 en índice Python

keep_modes = [
    i for i in range(n_modes)
    if i not in exclude_modes
]

fig, ax = plt.subplots(figsize=(8,5))

for i in keep_modes:

    ax.plot(
        f,
        psd_coefs_mean[i],
        label=f"Mode {i+1}",
        color=mode_colors[i],
        linewidth=2
    )

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD coefficient (a.u.)")

ax.set_xlim([f[0], f[-1]])

ax.legend()

fig.savefig(
    os.path.join(
        dynemo_plots_PSD_path,
        "group_mean_PSD_coefficients_without_mode2.png",
    )
)

cprint(
    f"   >>>     PSD coeficientes sin Mode 2 guardado en "
    f"{os.path.join(dynemo_plots_PSD_path, 'group_mean_PSD_coefficients_without_mode2.png')}"
)


############### PSD without mode 1 ###############

exclude_modes = [0]  # mode 1 en índice Python

keep_modes = [
    i for i in range(n_modes)
    if i not in exclude_modes
]

fig, ax = plt.subplots(figsize=(8,5))

for i in keep_modes:

    ax.plot(
        f,
        psd_coefs_mean[i],
        label=f"Mode {i+1}",
        color=mode_colors[i],
        linewidth=2
    )

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD coefficient (a.u.)")

ax.set_xlim([f[0], f[-1]])

ax.legend()

fig.savefig(
    os.path.join(
        dynemo_plots_PSD_path,
        "group_mean_PSD_coefficients_without_mode1.png",
    )
)

cprint(
    f"   >>>     PSD coeficientes sin Mode 1 guardado en "
    f"{os.path.join(dynemo_plots_PSD_path, 'group_mean_PSD_coefficients_without_mode1.png')}"
)



# Layout y títulos para las figuras combinadas (todos los modos juntos)
combined_n_rows = 2 if n_modes > 4 else 1
mode_titles = [f"Mode {i+1}" for i in range(n_modes)]
mask_file = os.path.join(paths.atlas_path, "MNI152_T1_8mm_brain.nii.gz")
parcellation_file = os.path.join(paths.atlas_path, "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz")


# 2. Plotear Power Maps
p = power.variance_from_spectra(f, psd_coefs)
cprint(f"   >>>     La forma de p {p.shape} deberia ser (subjects, modes, channels) \n  ")
mean_p = np.average(p, axis=0, weights=w)
cprint(f"   >>>     La forma de mean {mean_p.shape} deberia ser (modes, channels) \n  ")

# 2a. Mapas individuales: cada modo con su propia escala
power_map_path = os.path.join(dynemo_plots_power_map_path, "power_map.png")
power.save(
    mean_p,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
    filename=power_map_path,
)
cprint(f"   >>>     Power maps individuales guardados en {dynemo_plots_power_map_path}  \n  ")

# 2b. Figura combinada: todos los modos con escala compartida
power_demeaned = mean_p - mean_p.mean(axis=0, keepdims=True)
vmax_p = float(np.max(np.abs(power_demeaned)))
power_map_combined_path = os.path.join(dynemo_plots_power_map_path, "power_map_combined.png")
power.save(
    mean_p,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
    filename=power_map_combined_path,
    combined=True,
    titles=mode_titles,
    n_rows=combined_n_rows,
    plot_kwargs={"vmin": -vmax_p, "vmax": vmax_p},
)
cprint(f"   >>>     Power map combinado (escala compartida) guardado en {power_map_combined_path}  \n  ")



# 3. Visualizamos las coherence networks
cprint(f"   >>>     Deberia dar {coh.shape} = (subjects, modes, channels, channels, frequencies)  \n  ")
# Calcular la mean coherence en todas las frecuencias.
c = connectivity.mean_coherence_from_spectra(f, coh) 
cprint(f"   >>>     Deberia dar {c.shape} = (subjects, modes, channels, channels)  \n  ")
# Promedio over subjects
mean_c = np.average(c, axis=0, weights=w)
# Threshold the top 3% relative to the mean
thres_mean_c = connectivity.threshold(mean_c, percentile=97, subtract_mean=True)
cprint(f"   >>>     Visualizando las redes  \n  ")

# 3a. Redes individuales: cada modo con su propia escala de enlaces
coh_net_path = os.path.join(dynemo_plots_coherence_networks_path, "coherence_networks.png")
connectivity.save(
    thres_mean_c,
    parcellation_file=parcellation_file,
    plot_kwargs={"edge_cmap": "Reds"},
    filename=coh_net_path,
)
cprint(f"   >>>     Redes coherence individuales guardadas en {dynemo_plots_coherence_networks_path}\n  ")

# 3b. Figura combinada: todos los modos con escala de enlaces compartida
edge_vmax = float(np.max(thres_mean_c))
coh_net_combined_path = os.path.join(dynemo_plots_coherence_networks_path, "coherence_networks_combined.png")
connectivity.save(
    thres_mean_c,
    parcellation_file=parcellation_file,
    plot_kwargs={"edge_vmin": 0, "edge_vmax": edge_vmax, "edge_cmap": "Reds"},
    combined=True,
    titles=mode_titles,
    n_rows=combined_n_rows,
    filename=coh_net_combined_path,
)
cprint(f"   >>>     Redes coherence combinadas (escala compartida) guardadas en {coh_net_combined_path}\n  ")


# 4. Coherencia como mapa
cprint(f"   >>>     Visualizando las coherence maps  \n  ")
mean_c_map = connectivity.mean_connections(mean_c)

# 4a. Mapas individuales: cada modo con su propia escala
coh_map_path = os.path.join(dynemo_plots_coherence_maps_path, 'coherence_maps.png')
power.save(
    mean_c_map,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
    filename=coh_map_path,
)
cprint(f"   >>>     Coherence maps individuales guardados en {dynemo_plots_coherence_maps_path}\n  ")

# 4b. Figura combinada: todos los modos con escala compartida
cmap_demeaned = mean_c_map - mean_c_map.mean(axis=0, keepdims=True)
vmax_cm = float(np.max(np.abs(cmap_demeaned)))
coh_map_combined_path = os.path.join(dynemo_plots_coherence_maps_path, 'coherence_maps_combined.png')
power.save(
    mean_c_map,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    subtract_mean=True,
    filename=coh_map_combined_path,
    combined=True,
    titles=mode_titles,
    n_rows=combined_n_rows,
    plot_kwargs={"vmin": -vmax_cm, "vmax": vmax_cm},
)
cprint(f"   >>>     Coherence map combinado (escala compartida) guardado en {coh_map_combined_path}\n  ")


