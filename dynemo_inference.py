from osl_dynamics.models import load
import paths
import os
from osl_dynamics.data import Data
import pickle
import numpy as np




# Paths:
dynemo_generic_data = paths.dynemo_generic_data_path
dynemo_prepared_data_path = os.path.join(dynemo_generic_data, "prepared_data")
dynemo_object_data_path = os.path.join(dynemo_generic_data, "data_object")
dynemo_trained_data_path = os.path.join(dynemo_generic_data, "trained_model")
dynemo_infered_parameters_path = os.path.join(paths.dynemo_infered_results_path)
data_object_file = os.path.join(dynemo_object_data_path, "data.pkl")



#-----------------------------LOADING DATA-----------------------------#

print("\033[1;36m \n\n\n ---------------------------------------------- \033[0m")
print("\033[1;36m       Loading Trained Model and Data \033[0m")
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




#-----------------------------INFERED PARAMETERS-----------------------------#

print("\033[1;36m \n\n\n ---------------------------------------------- \033[0m")
print("\033[1;36m       Getting Infered Parameters \033[0m")
print("\033[1;36m ---------------------------------------------- \n \033[0m")

# In DyNeMo, alpha corresponds to the mixing coefficient time courses. 
# We can get these using the get_alpha method by passing the (prepared) training data.

# Get the alpha (a list of numpy arrays) for each subject
print("\033[1;36m   >>>     Obteniendo alphas\033[0m")
alpha = model.get_alpha(data_object)

print("\033[1;36m\n   ===== FORMATO DE ALPHAS =====\033[0m")
print(f"\033[1;36m   >>>     Tipo de dato: {type(alpha)}\033[0m")
print(f"\033[1;36m   >>>     Número de sujetos/sesiones: {len(alpha)}\033[0m")
print(f"\033[1;36m   >>>     Tipo de elementos: {type(alpha[0])}\033[0m")

print("\033[1;36m\n   ===== DIMENSIONES POR SUJETO =====\033[0m")
total_samples = 0
for i, alpha_subj in enumerate(alpha):
    print(f"\033[1;36m   Sujeto {i+1}: shape = {alpha_subj.shape} → (time_points={alpha_subj.shape[0]}, n_modes={alpha_subj.shape[1]})\033[0m")
    total_samples += alpha_subj.shape[0]

print(f"\033[1;36m\n   ===== VERIFICACIÓN TOTAL =====\033[0m")
print(f"\033[1;36m   >>>     Total de time points en alphas: {total_samples}\033[0m")
print(f"\033[1;36m   >>>     Total de samples en data_object: {data_object.n_samples}\033[0m")
diff = data_object.n_samples - total_samples
perc_diff = (diff / data_object.n_samples) * 100

if total_samples == data_object.n_samples:
    print("\033[1;32m   ✓ Las dimensiones coinciden exactamente\033[0m")
elif diff < data_object.n_samples * 0.01:  # Menos del 1% de diferencia
    print(f"\033[1;33m   ⚠ Diferencia: {diff} samples ({perc_diff:.3f}%)\033[0m")
    print("\033[1;32m   ✓ Diferencia esperada debido a sequence_length\033[0m")
else:
    print(f"\033[1;31m   ✗ ADVERTENCIA: Diferencia significativa de {diff} samples ({perc_diff:.2f}%)\033[0m")

# Save the infered parameters
print(f"\033[1;36m   >>>     Guardando los alpha en {dynemo_object_data_path} \033[0m")
os.makedirs(dynemo_infered_parameters_path, exist_ok=True)
pickle.dump(alpha, open(os.path.join(dynemo_infered_parameters_path, "alp.pkl"), "wb"))

# Get inferred state/mode means and covariances 
print("\033[1;36m   >>>     Obteniendo means y covariances\033[0m")
means, covs = model.get_means_covariances()

print(f"\033[1;36m   >>>     Guardando means y covariances en {dynemo_infered_parameters_path} \033[0m")
np.save(os.path.join(dynemo_infered_parameters_path, "means.npy"), means)
np.save(os.path.join(dynemo_infered_parameters_path, "covs.npy"), covs)

print("\033[1;36m\n   ✓ Parámetros inferidos obtenidos y guardados correctamente\033[0m")
