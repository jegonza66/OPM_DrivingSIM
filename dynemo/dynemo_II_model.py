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
import paths
from osl_dynamics.data import Data
from osl_dynamics.models.dynemo import Config
from osl_dynamics.models.dynemo import Model
import setup
from general_utility_functions import cprint, rprint, yprint, gprint
import matplotlib.pyplot as plt
import mne


# Setup
n_modes = 6
n_pca = 80
n_embeddings = 15
sequence_length = 100
force_retrain_model = True
exp_info = setup.exp_info()

# Paths:
dynemo_object_data_path = paths.dynemo_run_save_path(n_modes, n_embeddings, sequence_length, "DyNeMo_Object_Data")
dynemo_trained_data_path = paths.dynemo_run_save_path(n_modes, n_embeddings, sequence_length, "DyNeMo_Trained_Model")
dynemo_plots_training_path = paths.dynemo_run_plots_path(n_modes, n_embeddings, sequence_length, "Training")
os.makedirs(dynemo_object_data_path, exist_ok=True)
os.makedirs(dynemo_trained_data_path, exist_ok=True)
data_object_file =  os.path.join(dynemo_object_data_path, "data.pkl")
raw_data_file = os.path.join(dynemo_object_data_path, "raw_data.pkl")
force_retrain_model = True



#-----------------------------PREPARING DATA-----------------------------#

cprint(" \n\n\n ----------------------------------------------  ")
cprint("               Preparing Data  ")
cprint(" ---------------------------------------------- \n  ")

raw_sessions_data = []                 # {'code', 'data'} 

for subject_code in exp_info.subjects_ids:
    cprint(f"   >>>     Cargando datos para {subject_code} ")
    parc_file = os.path.join(paths.dynemo_preprocessing, subject_code,"parcellation", f"{subject_code}_parcel_data_spatial_basis_symmetric.npy")



    parc_ts = np.load(parc_file).T  # (time, parcels)

    # sfreq = 250
    # parc_ts = parc_ts.astype(np.float64)

    # parc_ts = mne.filter.filter_data(
    #     parc_ts.T,
    #     sfreq=sfreq,
    #     l_freq=1,
    #     h_freq=45,
    #     method='iir',
    #     verbose=False
    # ).T
    # parc_ts = parc_ts.astype(np.float32)
    # n_time, n_channels = parc_ts.shape
    # cprint(f"   >>>     Parcelado cargado para {subject_code}  ")
    # cprint(f"   >>>     Forma original: time={n_time}, channels={n_channels}  ")


    # Save original data  
    raw_sessions_data.append(parc_ts)

cprint("   >>>     Datos cargados correctamente. ")
cprint("   >>>     Sujetos totales: {len(raw_sessions_data)}")

for i, s in enumerate(raw_sessions_data):
    if s.shape[1] != 38:
        rprint(f"Advertencia: El sujeto {exp_info.subjects_ids[i]} tiene un número inesperado de canales ({s.shape[1]} en lugar de 38) ")

# Save prepared raw data
raw_data_obj = Data(raw_sessions_data)
raw_data_obj.save(raw_data_file)
cprint(f"   >>>     Raw data guardada en: {raw_data_file} ")


cprint("   >>>     Preparando los datos con osl_dynamics.data.Data ... ")
data = Data(raw_sessions_data)
methods = {
    "tde_pca": {"n_embeddings": n_embeddings, "n_pca_components": n_pca},
    "standardize": {},
}
data.prepare(methods)
data.save(data_object_file)
cprint(f"   >>>     Objeto Data guardado en: {dynemo_object_data_path} ")

# Print declarativo completo
cprint("\n   ===== INFORMACIÓN DEL DATASET ===== ")
cprint(f"   >>>     Número de sujetos/sesiones: {data.n_sessions} ")
cprint(f"   >>>     Orden de dimensiones esperado: (tiempo/muestras, canales/sensores) ")
cprint(f"   >>>     Total de muestras (todos los sujetos): {data.n_samples} ")
cprint(f"   >>>     Número de canales: {data.n_channels} ")


#-----------------------------BUILDING MODEL------------------------------------#

cprint(" \n\n\n ----------------------------------------------  ")
cprint("               Building Model  ")
cprint(" ---------------------------------------------- \n  ")

# Configure Dynemo parameters
cprint("   >>>     Configurando parámetros de DyNeMo ... ")
cprint("   >>>     To get more info about parameters you can access:  ")
cprint("           https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/3-3_dynemo_training.html ")
config = Config(
    n_modes=n_modes,
    n_channels=n_pca, # 30
    sequence_length=sequence_length,

    inference_n_units=128,
    inference_normalization="layer",
    model_n_units=128,
    model_normalization="layer",

    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,

    learn_means=False,
    learn_covariances=True,

    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=5,
    n_kl_annealing_epochs=20,

    batch_size=128,
    learning_rate=0.002,
    n_epochs=80,
)
cprint("   >>>     Parámetros de DyNeMo configurados: ")
print(config)

# Build model
cprint("   >>>     Construyendo modelo ")
model = Model(config)
model.summary()
cprint("   >>>     Modelo construido exitosamente ")


#-----------------------------TRAINING MODEL-----------------------------

cprint(" \n\n\n ----------------------------------------------  ")
cprint("               Training Model  ")
cprint(" ---------------------------------------------- \n  ")

cprint("   >>>     Esta parte podría demorar...  ")

if force_retrain_model or not os.path.exists(dynemo_trained_data_path):
    cprint(f"   >>>     No existe el modelo entrenado o se activo force_retrain_model, iniciando entrenamiento...  ")
    # Train the model for a short period on a small random subset of the data
    cprint(f"   >>>     Buscando buena inicialización  ")
    init_history = model.random_subset_initialization(
        data,
        n_epochs=5,
        n_init=10,
        take=0.25,
    )

    # After good initialization, we do the full training of the model.
    # save_best_after keeps the lowest-loss weights, but only AFTER KL annealing
    # finishes (so it ignores the misleadingly low pre-annealing loss).
    cprint(f"   >>>     Entrenando el modelo DyNeMo completo ...  ")
    history = model.fit(data, save_best_after=config.n_kl_annealing_epochs)

    # Save the trained model
    model.save(dynemo_trained_data_path)
    cprint(f"   >>>     Modelo entrenado guardado correctamente en: {dynemo_trained_data_path} ...  ") 

else:
    cprint(f"   >>>     Modelo entrenado ya existe, cargándolo desde el disco...  ")
    model = Model.load(dynemo_trained_data_path)
    cprint(f"   >>>     Modelo entrenado cargado correctamente desde: {dynemo_trained_data_path} ...  ")

# Verify that modes did not collapse
alpha = model.get_alpha(data)
for i, a in enumerate(alpha):
    fo = a.mean(axis=0)
    rprint(f"Sujeto {i+1}: fractional occupancy = {fo.round(3)} ")

# Plot training loss
plt.figure()
plt.plot(history["loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss")
os.makedirs(dynemo_plots_training_path, exist_ok=True)
plt.savefig(os.path.join(dynemo_plots_training_path, "training_loss.png"))
plt.show()