import os
import numpy as np
import paths
from osl_dynamics.data import Data
from osl_dynamics.models.dynemo import Config
from osl_dynamics.models.dynemo import Model
import zipfile

# Paths:
dynemo_generic_data = paths.dynemo_generic_data_path
dynemo_object_data_path = os.path.join(dynemo_generic_data, "data_object")
os.makedirs(dynemo_object_data_path, exist_ok=True)
dynemo_trained_data_path = os.path.join(dynemo_generic_data, "trained_model")
os.makedirs(dynemo_trained_data_path, exist_ok=True)
data_object_file = os.path.join(dynemo_object_data_path, "data.pkl")
dynemo_prepared_data_path = os.path.join(dynemo_generic_data, "prepared_data")
os.makedirs(dynemo_prepared_data_path, exist_ok=True)



#-----------------------------PREPARING DATA-----------------------------#

print("\033[1;36m \n\n\n ---------------------------------------------- \033[0m")
print("\033[1;36m               Preparing Data \033[0m")
print("\033[1;36m ---------------------------------------------- \n \033[0m")


# Download the dataset from OSL using osfclient
def get_data(name, rename=None):
    if rename is None:
        rename = name

    # Download from OSF
    zip_filename = f"{name}.zip"
    os.system(f"osf -p by2tc fetch data/{name}.zip")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(dynemo_prepared_data_path)
    os.remove(zip_filename)

    return f"Data downloaded to: {dynemo_prepared_data_path}"

# Download the dataset (approximately 21 MB)
get_data("notts_mrc_meguk_giles_5_subjects")
print("\033[1;36m   >>>     Dataset descargado correctamente\033[0m")



# Load the prepared data with osl_dynamics.data.Data class
print("\033[1;36m   >>>     Preparando los datos con osl_dynamics.data.Data ...\033[0m")
# 'prepared_data' contains the data in (n_samples, n_channels) format
data_obj = Data(dynemo_prepared_data_path)
data_obj.save(data_object_file)
print(f"\033[1;36m   >>>     Objeto Data guardado en: {dynemo_object_data_path}\033[0m")
# Print declarativo completo
print(f"\033[1;36m\n   ===== INFORMACIÓN DEL DATASET =====\033[0m")
print(f"\033[1;36m   >>>     Número de sujetos/sesiones: {data_obj.n_sessions}\033[0m")
print(f"\033[1;36m   >>>     Número de canales: {data_obj.n_channels}\033[0m")
print(f"\033[1;36m   >>>     Total de muestras (todos los sujetos): {data_obj.n_samples}\033[0m")

print(f"\033[1;36m\n   ===== FORMATO =====\033[0m")
print(f"\033[1;36m   >>>     Orden de dimensiones: (samples, channels) ✓\033[0m")
print(f"\033[1;36m   >>>     Primera dimensión = tiempo/muestras\033[0m")
print(f"\033[1;36m   >>>     Segunda dimensión = canales/sensores\033[0m")




#-----------------------------BUILDING MODEL------------------------------------#

print("\033[1;36m \n\n\n ---------------------------------------------- \033[0m")
print("\033[1;36m               Building Model \033[0m")
print("\033[1;36m ---------------------------------------------- \n \033[0m")

# Configure Dynemo parameters
print("\033[1;36m   >>>     Configurando parámetros de DyNeMo ...\033[0m")
print("\033[1;36m   >>>     To get more info about parameters you can access: \033[0m")
print("\033[1;36m           https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/3-3_dynemo_training.html\033[0m")
config = Config(
    n_modes=7,
    n_channels=38, # Adjust this based on your data's number of channels, in 5 subjects dataset we have 38 channels
    sequence_length=100,
    inference_n_units=64,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=5,
    n_kl_annealing_epochs=10,
    batch_size=32,
    learning_rate=0.01,
    n_epochs=20,
)
print("\033[1;36m   >>>     Parámetros de DyNeMo configurados:\033[0m")
print(config)


# Build model
model = Model(config)
model.summary()




#-----------------------------TRAINING MODEL-----------------------------

print("\033[1;36m \n\n\n ---------------------------------------------- \033[0m")
print("\033[1;36m               Training Model \033[0m")
print("\033[1;36m ---------------------------------------------- \n \033[0m")

print("\033[1;36m   >>>     Esta parte podría demorar... \033[0m")

# Train the model for a short period on a small random subset of the data
print("\033[1;36m   >>>     Buscando buena inicialización \033[0m")
init_history = model.random_subset_initialization(data_obj, n_epochs=2, n_init=5, take=0.25)

# After good initialization, we do the full training of the model
print("\033[1;36m   >>>     Entrenando el modelo DyNeMo completo ... \033[0m")
history = model.fit(data_obj)

# Save the trained model
model.save(dynemo_trained_data_path)
print("\033[1;36m   >>>     Modelo entrenado guardado correctamente en: {dynemo_trained_data_path} ... \033[0m") 

