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

import gc
import numpy as np
import mne
import mne.beamformer as beamformer
import paths
import load
import setup
from general_utility_functions import cprint, rprint, yprint

from dynemo__utility_functions import (parcellate_spatial_basis_symmetric, save_parcel_data_as_fif, 
            get_meg_ch_types, preprocess_raw_for_dynemo, save_coreg_qc, make_or_load_volume_src,
            make_or_load_forward, make_or_load_data_cov, make_or_load_lcmv,
            apply_lcmv_raw, get_voxel_coords_mni)


# ============================================================
# DYNEMO PREPROCESSING - SOURCE LOCALIZATION PIPELINE
# raw -> filter/downsample -> volume LCMV -> voxel data -> spatial_basis parcellation
# ============================================================

exp_info = setup.exp_info()

# ----------------------------
# Set up
# ----------------------------
RESAMPLE_FREQ = 250
BANDPASS = (1, 45)
POS_MM = 8.0
REG = 0.05

# LCMV orientation
PICK_ORI = "max-power"

# Data
DATA_TYPE = "processed"
COV_TAG = "cov1-45hz"

# BEM ico spacing (matches sourcemodel_setup.py)
BEM_SPACING = "ico4"

# Atlas
PARCELLATION_FILE = os.path.join(paths.atlas_path, "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",)

# FreeSurfer reconstructions live in DATA/MRI/freesurfer
SUBJECTS_DIR = os.path.join(paths.mri_path, "freesurfer")
os.environ["SUBJECTS_DIR"] = SUBJECTS_DIR

OUT_ROOT = paths.dynemo_preprocessing




# ============================================================
# Main loop
# ============================================================

for subject_id in exp_info.subjects_ids:

    subject_code = subject_id

    cprint("\n" + "=" * 80)
    cprint(f"EMPEZANDO DYNEMO PREPROCESSING PARA {subject_code}")
    cprint("=" * 80)

    subj_dir = os.path.join(OUT_ROOT, subject_code)
    os.makedirs(subj_dir, exist_ok=True)

    preproc_dir = os.path.join(subj_dir, "preprocessed")
    os.makedirs(preproc_dir, exist_ok=True)
    coreg_dir = os.path.join(subj_dir, "coreg")
    os.makedirs(coreg_dir, exist_ok=True)
    src_dir = os.path.join(subj_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    fwd_dir = os.path.join(subj_dir, "forward")
    os.makedirs(fwd_dir, exist_ok=True)
    cov_dir = os.path.join(subj_dir, "covariance")
    os.makedirs(cov_dir, exist_ok=True)
    lcmv_dir = os.path.join(subj_dir, "lcmv")
    os.makedirs(lcmv_dir, exist_ok=True)
    voxel_dir = os.path.join(subj_dir, "voxel")
    os.makedirs(voxel_dir, exist_ok=True)
    parc_dir = os.path.join(subj_dir, "parcellation")
    os.makedirs(parc_dir, exist_ok=True)
    qc_dir = os.path.join(paths.dynemo_plots_preprocessing_path, subject_code, "qc")
    os.makedirs(qc_dir, exist_ok=True)

    subject = setup.subject(subject_id=subject_id)

    # All participants have MRI -> use their own FreeSurfer reconstruction
    mri_subject = subject_id

    # ----------------------------
    # Load processed MEG data (ICA + further processing), as in other modules
    # ----------------------------
    raw = load.meg(subject_id=subject_id, meg_params={"data_type": DATA_TYPE}).load_data()
    raw = raw.copy()
    raw = raw.pick("mag")

    ch_types = get_meg_ch_types(raw)
    cprint(f"Processed data cargada para {subject_code}")
    cprint(f"Canales usados: {ch_types}")


    # ----------------------------
    # Existing trans + BEM
    # ----------------------------
    trans_path = os.path.join(SUBJECTS_DIR, subject_id, "bem", f"{subject_id}-trans.fif")
    trans = mne.read_trans(trans_path)
    cprint(f"TRANS cargado para {subject_code}")

    # Reuse the project's BEM (built in sourcemodel_setup.py); compute if missing
    bem_path = os.path.join(paths.sources_path + subject_code, f"{subject_code}_bem_{BEM_SPACING}-sol.fif")
    if os.path.exists(bem_path):
        bem = mne.read_bem_solution(bem_path)
        cprint(f"BEM cargado para {subject_code}")
    else:
        cprint(f"No se encontró BEM para {subject_code}, calculando ({BEM_SPACING})...")
        bem_model = mne.make_bem_model(subject=mri_subject, ico=int(BEM_SPACING[-1]),
                                       conductivity=[0.3], subjects_dir=SUBJECTS_DIR)
        bem = mne.make_bem_solution(bem_model)
        os.makedirs(os.path.dirname(bem_path), exist_ok=True)
        mne.write_bem_solution(bem_path, bem, overwrite=True)

    save_coreg_qc(
        raw=raw,
        trans=trans,
        subject_code=subject_code,
        mri_subject=mri_subject,
        out_png=os.path.join(qc_dir, f"{subject_code}_coreg_alignment.png"),
        subjects_dir=SUBJECTS_DIR,
    )
    cprint(f"Coreg QC guardado para {subject_code} en {qc_dir}")
    yprint(f"Revisar que la alineación entre sensores y MRI es correcta.")

    # ----------------------------
    # Preprocess raw 
    # ----------------------------
    preproc_fif = os.path.join(preproc_dir, f"{subject_code}_preproc_1-45Hz_250Hz-raw.fif")

    raw_preproc = preprocess_raw_for_dynemo(raw, preproc_fif, BANDPASS[0], BANDPASS[1], RESAMPLE_FREQ)

    rprint(f"Datos preprocesados: {raw_preproc.get_data(picks='meg').shape}")
    rprint(f"Filtrado desde {raw_preproc.info['highpass']} Hz hasta {raw_preproc.info['lowpass']} Hz")


    # ----------------------------
    # Rank
    # ----------------------------
    rank = mne.compute_rank(raw, rank=None)
    cprint(f"Rank estimado por MNE: {rank}")

    # ----------------------------
    # Volume source space, forward model
    # ----------------------------
    src_file = os.path.join(src_dir, f"{subject_code}_volume_pos{int(POS_MM)}mm-src.fif",)

    fwd_file = os.path.join(fwd_dir, f"{subject_code}_volume_pos{int(POS_MM)}mm-fwd.fif",)

    src = make_or_load_volume_src(subject_code=subject_code, mri_subject=mri_subject, bem=bem, 
                                  out_file=src_file, subjects_dir=SUBJECTS_DIR, pos_mm=POS_MM)

    fwd = make_or_load_forward(raw=raw_preproc, trans=trans, src=src, bem=bem, out_file=fwd_file,)
    cprint(f"Source space cargado para {subject_code}.")
    cprint(f"Forward model cargado para {subject_code}.")

    # ----------------------------
    # Covariances
    # ----------------------------
    # Like the other source modules in this project, LCMV uses only the data
    # covariance + regularisation (no empty-room noise covariance available).
    noise_cov = None

    data_cov_file = os.path.join(cov_dir, f"{subject_code}_data_{COV_TAG}-cov.fif",)

    data_cov = make_or_load_data_cov(raw=raw_preproc, rank=rank, out_file=data_cov_file,)
    cprint(f"Covarianza de datos cargada para {subject_code}.")

    # ----------------------------
    # LCMV filters
    # ----------------------------
    lcmv_file = os.path.join(lcmv_dir,f"{subject_code}_volume_pos{int(POS_MM)}mm_{PICK_ORI}_{COV_TAG}-lcmv.fif",)

    filters = make_or_load_lcmv(raw=raw_preproc, fwd=fwd, data_cov=data_cov, noise_cov=noise_cov, rank=rank,
                                out_file=lcmv_file, beamformer=beamformer, pick_ori=PICK_ORI, reg=REG)
    cprint(f"Filtros LCMV cargados para {subject_code}.")

    # ----------------------------
    # Apply LCMV to raw -> voxel data
    # ----------------------------
    voxel_data_file = os.path.join( voxel_dir, f"{subject_code}_voxel_data_volume_pos{int(POS_MM)}mm_{PICK_ORI}_{COV_TAG}.npy",)

    voxel_data = apply_lcmv_raw(raw=raw_preproc, filters=filters, out_npy=voxel_data_file, beamformer=beamformer,)
    cprint(f"LCMV aplicado a raw para {subject_code}, datos de voxel guardados en {voxel_data_file}.")
    cprint(f"Voxel data shape: {voxel_data.shape}  # voxels x time")

    # ----------------------------
    # Voxel coords in MNI mm
    # ----------------------------
    voxel_coords_file = os.path.join(voxel_dir, f"{subject_code}_voxel_coords_mni_volume_pos{int(POS_MM)}mm.npy",)

    voxel_coords_mni = get_voxel_coords_mni(fwd=fwd, trans=trans, subject_code=subject_code,
                                            mri_subject=mri_subject, subjects_dir=SUBJECTS_DIR, out_file=voxel_coords_file,)

    cprint(f"Voxel coords shape: {voxel_coords_mni.shape}  # voxels x 3, MNI mm")

    # ----------------------------
    # Parcellation: spatial_basis + symmetric
    # ----------------------------
    parc_npy = os.path.join(parc_dir, f"{subject_code}_parcel_data_spatial_basis_symmetric.npy",)

    parc_fif = os.path.join(parc_dir,f"{subject_code}_lcmv-parc-raw.fif",)

    parcel_data, voxel_weightings, parcellation_asmatrix = (
        parcellate_spatial_basis_symmetric(voxel_data=voxel_data, voxel_coords_mni=voxel_coords_mni,parcellation_file=PARCELLATION_FILE,)
    )

    np.save(parc_npy, parcel_data)

    np.save( os.path.join(parc_dir, f"{subject_code}_voxel_weightings_spatial_basis.npy"), voxel_weightings,)

    np.save(os.path.join(parc_dir, f"{subject_code}_parcellation_asmatrix.npy"),parcellation_asmatrix,)

    save_parcel_data_as_fif(parcel_data=parcel_data, raw=raw_preproc, filename=parc_fif, extra_chans="stim",)

    cprint(f"Parcel data shape: {parcel_data.shape}  # parcels x time")
    cprint(f"Parcel FIF final: {parc_fif}")

    # ----------------------------
    # Summary file
    # ----------------------------
    summary_file = os.path.join(
        subj_dir,
        f"{subject_code}_dynemo_preprocessing_summary.txt",
    )

    with open(summary_file, "w") as f:
        f.write(f"Subject: {subject_code}\n")
        f.write(f"Output dir: {subj_dir}\n")
        f.write(f"Input data type: {DATA_TYPE}\n")
        f.write(f"Bandpass: {BANDPASS[0]}-{BANDPASS[1]} Hz\n")
        f.write(f"Resample: {RESAMPLE_FREQ} Hz\n")
        f.write("Source space: volume\n")
        f.write(f"Grid spacing: {POS_MM} mm\n")
        f.write(f"LCMV pick_ori: {PICK_ORI}\n")
        f.write(f"LCMV reg: {REG}\n")
        f.write(f"Rank: {rank}\n")
        f.write(f"Trans path: {trans_path}\n")
        f.write(f"BEM path: {bem_path}\n")
        f.write(f"Source space file: {src_file}\n")
        f.write(f"Forward file: {fwd_file}\n")
        f.write(f"Data covariance file: {data_cov_file}\n")
        f.write(f"LCMV file: {lcmv_file}\n")
        f.write(f"Voxel data file: {voxel_data_file}\n")
        f.write(f"Voxel coords MNI file: {voxel_coords_file}\n")
        f.write(f"Parcellation file: {PARCELLATION_FILE}\n")
        f.write(f"Parcel npy: {parc_npy}\n")
        f.write(f"Parcel fif: {parc_fif}\n")
        f.write(f"Voxel data shape: {voxel_data.shape}\n")
        f.write(f"Voxel coords shape: {voxel_coords_mni.shape}\n")
        f.write(f"Parcel data shape: {parcel_data.shape}\n")

    cprint(f"Summary guardado: {summary_file}")

    del raw, raw_preproc, src, fwd, filters, noise_cov, data_cov
    del voxel_data, voxel_coords_mni, parcel_data
    gc.collect()

cprint("\nDYNEMO PREPROCESSING TERMINADO PARA TODOS LOS SUJETOS.")