import os

# Anchor paths to the location of this file (always inside the Scripts folder),
# so modules work regardless of the current working directory (e.g. when run
# from a subfolder such as Scripts/dynemo).
scripts_path = os.path.dirname(os.path.realpath(__file__))
main_path = os.path.dirname(scripts_path) + os.sep

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
exp_video_path = main_path + 'DATA/EXP/'
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

# ----------------------------------------------------------------------------
# DyNeMo pipeline paths
# ----------------------------------------------------------------------------
# Atlas / parcellation niftis (e.g. fmri_d100..._ds8mm.nii.gz, MNI152_T1_8mm_brain.nii.gz)
atlas_path = os.path.join(mri_path, 'atlases')

# Event/correlation tables used by the temporal analysis (script V)
correlation_path = main_path + 'Save/Correlation/'

# DyNeMo root and derived data/model paths
dynemo_path = os.path.join(main_path, "Save/DyNeMo")
dynemo_preprocessing = os.path.join(dynemo_path, "DyNeMo_Preprocessing")
dynemo_object_data_path = os.path.join(dynemo_path, "DyNeMo_Object_Data")
dynemo_prepared_data_path = os.path.join(dynemo_path, "DyNeMo_Prepared_Data")
dynemo_trained_data_path = os.path.join(dynemo_path, "DyNeMo_Trained_Model")
dynemo_infered_parameters_path = os.path.join(dynemo_path, "DyNeMo_Infered_Parameters")
dynemo_temporal_analysis_path = os.path.join(dynemo_path, "DyNeMo_Temporal_Analysis")
# Event-locked TRF of the DyNeMo mixing coefficients (script VI)
dynemo_mixing_trf_path = os.path.join(dynemo_path, "DyNeMo_Mixing_TRF")

# DyNeMo plot paths
dynemo_plots_path = os.path.join(plots_path, "DyNeMo")
dynemo_plots_PSD_path = os.path.join(dynemo_plots_path, "PSD")
dynemo_plots_power_map_path = os.path.join(dynemo_plots_path, "Power_Maps")
dynemo_plots_coherence_networks_path = os.path.join(dynemo_plots_path, "Coherence_Networks")
dynemo_plots_coherence_maps_path = os.path.join(dynemo_plots_path, "Coherence_Maps")
dynemo_plots_mixing_coefficients_path = os.path.join(dynemo_plots_path, "Mixing_Coefficients")
dynemo_plots_training_path = os.path.join(dynemo_plots_path, "Training")
dynemo_plots_temporal_analysis_path = os.path.join(dynemo_plots_path, "Temporal_Analysis")
dynemo_plots_preprocessing_path = os.path.join(dynemo_plots_path, "Preprocessing")
dynemo_plots_mixing_trf_path = os.path.join(dynemo_plots_path, "Mixing_TRF")


# ----------------------------------------------------------------------------
# DyNeMo per-run paths
# ----------------------------------------------------------------------------
# A DyNeMo "run" is identified by the key parameters that change the prepared
# data and the trained model (number of time-delay embeddings and the model
# sequence length). Tagging the output folders with these parameters keeps the
# plots and saved results of different runs side by side instead of
# overwriting each other (e.g. "emb15_seq200").
def dynemo_run_tag(n_modes, n_embeddings, sequence_length):
    """Folder name identifying a DyNeMo run by its key parameters."""
    return f"modes{n_modes}_emb{n_embeddings}_seq{sequence_length}"


def dynemo_run_plots_path(n_modes, n_embeddings, sequence_length, subdir=None):
    """Per-run plots folder, e.g. Plots/DyNeMo/emb15_seq200[/<subdir>]."""
    base = os.path.join(dynemo_plots_path, dynemo_run_tag(n_modes, n_embeddings, sequence_length))
    return base if subdir is None else os.path.join(base, subdir)


def dynemo_run_save_path(n_modes, n_embeddings, sequence_length, subdir=None):
    """Per-run Save folder, e.g. Save/DyNeMo/emb15_seq200[/<subdir>]."""
    base = os.path.join(dynemo_path, dynemo_run_tag(n_modes, n_embeddings, sequence_length))
    return base if subdir is None else os.path.join(base, subdir)

