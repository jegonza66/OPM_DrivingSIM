import os

scripts_path = os.path.dirname(os.path.realpath(__name__))
main_path = scripts_path.replace('Scripts', '')

# Check that DATA folder exists in main_path
data_path_exists = os.path.exists(main_path + 'DATA/')
if not data_path_exists:
    raise AssertionError(f'DATA/ folder not found in main path: {main_path}\n'
                         f'Please refer to https://github.com/jegonza66/MEGEYEDYN and copy the directory structure from Readme.md')

opm_path = main_path + 'DATA/OPM/'
et_path = main_path + 'DATA/ET/'
mri_path = main_path + 'DATA/MRI/'
opt_path = main_path + 'DATA/OPT/'
exp_path = main_path + 'DATA/Experiment/'


save_path = main_path + 'Save/'
os.makedirs(save_path, exist_ok=True)

preproc_path = main_path + 'Save/Preproc_Data/'
os.makedirs(preproc_path, exist_ok=True)

filtered_path_raw = main_path + 'Save/Filtered_Data_RAW/'
os.makedirs(filtered_path_raw, exist_ok=True)

filtered_path_ica = main_path + 'Save/Filtered_Data_ICA/'
os.makedirs(filtered_path_ica, exist_ok=True)

ica_path = main_path + 'Save/ICA_Data/'
os.makedirs(ica_path, exist_ok=True)

plots_path = main_path + 'Plots/'
os.makedirs(plots_path, exist_ok=True)

sources_path = main_path + 'Save/Source_Data/'
os.makedirs(sources_path, exist_ok=True)
