import os

scripts_path = os.path.dirname(os.path.realpath(__name__))
main_path = scripts_path.replace('Scripts', '')

# Check that DATA folder exists in main_path
data_path_exists = os.path.exists(main_path + 'DATA/')
if not data_path_exists:
    raise AssertionError(f'DATA/ folder not found in main path: {main_path}\n'
                         f'Please refer to https://github.com/jegonza66/OPM_DrivingSIM and copy the directory structure from Readme.md')

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

tsss_raw_path = main_path + 'Save/tsss_raw_Data/'
os.makedirs(tsss_raw_path, exist_ok=True)

tsss_raw_annot_path = main_path + 'Save/tsss_raw_Data_annot/'
os.makedirs(tsss_raw_annot_path, exist_ok=True)

filtered_path_tsss = main_path + 'Save/Filtered_Data_tsss/'
os.makedirs(filtered_path_tsss, exist_ok=True)

filtered_path_tsss_annot = main_path + 'Save/Filtered_Data_tsss_annot/'
os.makedirs(filtered_path_tsss_annot, exist_ok=True)

ica_path = main_path + 'Save/ICA_Data/'
os.makedirs(ica_path, exist_ok=True)

ica_annot_path = main_path + 'Save/ICA_Data_annot/'
os.makedirs(ica_path, exist_ok=True)

filtered_path_ica = main_path + 'Save/Filtered_Data_ICA/'
os.makedirs(filtered_path_ica, exist_ok=True)

filtered_path_ica_annot = main_path + 'Save/Filtered_Data_ICA_annot/'
os.makedirs(filtered_path_ica, exist_ok=True)

plots_path = main_path + 'Plots/'
os.makedirs(plots_path, exist_ok=True)

sources_path = main_path + 'Save/Source_Data/'
os.makedirs(sources_path, exist_ok=True)
