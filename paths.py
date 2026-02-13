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
bh_path = main_path + 'DATA/BEHAV/'


save_path = main_path + 'Save/'

processed_path = main_path + 'Save/Processed_Data/'

processed_path_annot = main_path + 'Save/Processed_Data_annot/'

filtered_path_processed = main_path + 'Save/Filtered_Data_Processed/'

filtered_path_processed_annot = main_path + 'Save/Filtered_Data_Processed_annot/'

filtered_path_raw = main_path + 'Save/Filtered_Data_RAW/'

tsss_raw_path = main_path + 'Save/tsss_raw_Data/'

tsss_raw_annot_path = main_path + 'Save/tsss_raw_Data_annot/'

filtered_path_tsss = main_path + 'Save/Filtered_Data_tsss/'

filtered_path_tsss_annot = main_path + 'Save/Filtered_Data_tsss_annot/'

ica_path = main_path + 'Save/ICA_Data/'

ica_annot_path = main_path + 'Save/ICA_Data_annot/'

filtered_path_ica = main_path + 'Save/Filtered_Data_ICA/'

filtered_path_ica_annot = main_path + 'Save/Filtered_Data_ICA_annot/'

plots_path = main_path + 'Plots/'

sources_path = main_path + 'Save/Source_Data/'

dynemo_generic_data_path = os.path.join(main_path, "Save/DyNeMo/DyNeMo_Generic_Data")

dynemo_infered_results_path = os.path.join(main_path, "Save/DyNeMo/DyNeMo_Infered_Results")

dynemo_spectra_path = os.path.join(main_path, "Save/DyNeMo/DyNeMo_Spectra")