import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy
import scipy.spatial
import scipy.sparse.linalg
import nibabel as nib
import mne
from general_utility_functions import cprint, rprint, yprint
from mne.transforms import apply_trans
from osl_dynamics.data import Data
import paths
import setup




# ============================================================
# Dynemo utility functions
# ============================================================

# Calcula exactamente cuántos samples fueron eliminados al inicio entre raw_data.pkl y trimmed_data para un sujeto.
def get_subject_trim_start(subject_code, n_embeddings=15, sequence_length=100, max_search=5000):

    exp_info = setup.exp_info()
    raw_data_file = os.path.join( paths.dynemo_object_data_path, "raw_data.pkl")
    data = Data(raw_data_file)
    raw_ts = data.time_series()

    # Aplicar mismo corte que se utiliza en Regression Spectra
    trimmed_ts = data.trim_time_series(n_embeddings=n_embeddings, sequence_length=sequence_length, )
    subject_idx = exp_info.subjects_ids.index(subject_code)

    x_raw = raw_ts[subject_idx]
    x_trim = trimmed_ts[subject_idx]

    first_trim = x_trim[0]

    d_start = np.linalg.norm(x_raw[:max_search] - first_trim,axis=1, )
    trim_start_exact = int(np.argmin(d_start))

    return trim_start_exact


# Function to get event rows for button presses or fixations
def get_event_rows(df, event_type):
    label = df["label"].astype(str).str.lower().str.strip()

    if event_type == "button":
        return df[label.str.contains("red|blue", na=False)].copy()

    if event_type == "fixation":
        return df[df["event_type"] == "fix"].copy()

    if event_type == "hazard_fixation":
        return df[label.str.contains("red|blue", na=False)].copy()

    return None

# Function to find the nearest kept_time for each event time, to align them
# Returns the indices of the nearest kept_times for each event_time
def nearest_kept_indices(event_times, kept_times):
    idx = np.searchsorted(kept_times, event_times) 
    idx = np.clip(idx, 1, len(kept_times) - 1) 

    left = idx - 1
    right = idx

    choose_right = (
        np.abs(kept_times[right] - event_times)
        <
        np.abs(kept_times[left] - event_times)
    )

    nearest = left.copy()
    nearest[choose_right] = right[choose_right]

    return nearest

# Function to recover kept_times file, which has the times of the samples that were kept after preprocessing
def find_kept_times_file(subject_code, dynemo_preprocessing_dir):
    voxel_dir = os.path.join( dynemo_preprocessing_dir, subject_code, "voxel")

    if not os.path.exists(voxel_dir):
        return None

    kept_files = [
        f for f in os.listdir(voxel_dir)
        if f.endswith("_kept_times.npy")
    ]

    if len(kept_files) == 0:
        rprint(f">>> No se encontró ningún archivo _kept_times.npy para {subject_code} en {voxel_dir}")
        return None

    if len(kept_files) > 1:
        yprint(f">>> Hay varios kept_times para {subject_code}. Uso el primero:")
        for f in kept_files:
            yprint(f"    {f}")

    return os.path.join(voxel_dir, kept_files[0])


def get_meg_ch_types(raw):
    ch_types = sorted(set(raw.copy().pick_types(meg=True).get_channel_types()))

    has_mag = "mag" in ch_types
    has_grad = "grad" in ch_types

    if has_mag and has_grad:
        return ["mag", "grad"]
    if has_mag:
        return ["mag"]
    if has_grad:
        return ["grad"]

    raise RuntimeError("No encontré canales MEG en el raw.")


def preprocess_raw_for_dynemo(raw, out_file, l_freq, h_freq, resample_freq):
    raw = raw.copy()
    raw.load_data()

    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params={"order": 5, "ftype": "butter"},
    )

    raw.resample(resample_freq)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    raw.save(out_file, overwrite=True)

    return raw



def save_coreg_qc(raw, trans, subject_code, mri_subject, out_png, subjects_dir):
    try:
        fig = mne.viz.plot_alignment(
            raw.info,
            trans=trans,
            subject=mri_subject,
            subjects_dir=subjects_dir,
            surfaces=dict(brain=0.7, head=0.4),
            dig=True,
            eeg=False,
            meg="sensors",
            show_axes=True,
        )

        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.plotter.screenshot(out_png)
        fig.plotter.close()

    except Exception as exc:
        yprint(f"No pude guardar QC de coreg para {subject_code}: {exc}")


def make_or_load_volume_src(subject_code, mri_subject, bem, out_file, subjects_dir, pos_mm=8):
    if os.path.exists(out_file):
        cprint(f"Cargando source space volumétrico: {out_file}")
        return mne.read_source_spaces(out_file)

    cprint(f"Creando source space volumétrico {pos_mm} mm para {subject_code}")

    src = mne.setup_volume_source_space(
        subject=mri_subject,
        subjects_dir=subjects_dir,
        bem=bem,
        pos=pos_mm,
        sphere_units="mm",
        add_interpolator=True,
    )

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    mne.write_source_spaces(out_file, src, overwrite=True)

    return src

def make_or_load_forward(raw, trans, src, bem, out_file):
    if os.path.exists(out_file):
        cprint(f"Cargando forward existente: {out_file}")
        return mne.read_forward_solution(out_file)

    cprint("Creando forward model volumétrico completo")

    fwd = mne.make_forward_solution(
        info=raw.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=5.0,
    )

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    mne.write_forward_solution(out_file, fwd, overwrite=True)

    return fwd


def make_or_load_data_cov(raw, rank, out_file):
    if os.path.exists(out_file):
        cprint(f"Cargando data covariance: {out_file}")
        return mne.read_cov(out_file)

    cprint("Calculando data covariance sobre raw filtrado 1-45 Hz / 250 Hz")

    data_cov = mne.compute_raw_covariance(
        raw,
        method="empirical",
        rank=rank,
        reject=None,
    )

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    mne.write_cov(out_file, data_cov, overwrite=True)

    return data_cov



def make_or_load_lcmv(raw, fwd, data_cov, noise_cov, rank, out_file, beamformer, pick_ori, reg):
    if os.path.exists(out_file):
        cprint(f"Cargando filtros LCMV: {out_file}")
        return beamformer.read_beamformer(out_file)

    cprint("Creando filtros LCMV volumétricos")

    filters = beamformer.make_lcmv(
        info=raw.info,
        forward=fwd,
        data_cov=data_cov,
        reg=reg,
        noise_cov=noise_cov,
        pick_ori=pick_ori,
        rank=rank,
        weight_norm="unit-noise-gain",
    )

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    filters.save(out_file, overwrite=True)

    return filters


def make_raw_without_bad_annotations(raw):
    """
    OSL source_recon.apply_lcmv_beamformer excludes bad segments.
    This function creates a continuous Raw object containing only samples
    not rejected by annotations, so LCMV is applied to clean data only.
    """
    raw_meg = raw.copy().pick_types(meg=True, eeg=False, stim=False, exclude=[])

    data, times = raw_meg.get_data(
        reject_by_annotation="omit",
        return_times=True,
    )

    info = raw_meg.info.copy()
    raw_clean = mne.io.RawArray(data, info)

    raw_clean.set_meas_date(raw.info["meas_date"])

    return raw_clean, times


def apply_lcmv_raw(raw, filters, out_npy, beamformer):
    """
    Bibliography-aligned: apply LCMV once, no chunking.
    Bad annotated samples are omitted before applying LCMV.
    """
    if os.path.exists(out_npy):
        cprint(f"Cargando voxel_data existente: {out_npy}")
        return np.load(out_npy)

    cprint("Aplicando LCMV al raw completo, sin chunks")

    raw_clean, kept_times = make_raw_without_bad_annotations(raw)

    stc = beamformer.apply_lcmv_raw(
        raw=raw_clean,
        filters=filters,
    )

    voxel_data = stc.data.astype(np.float32)

    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, voxel_data)

    kept_times_file = out_npy.replace(".npy", "_kept_times.npy")
    np.save(kept_times_file, kept_times)

    return voxel_data



def get_voxel_coords_mni(fwd, trans, subject_code, mri_subject, out_file, subjects_dir):
    if os.path.exists(out_file):
        cprint(f"Cargando voxel_coords MNI: {out_file}")
        return np.load(out_file)

    # Coordenadas de las fuentes del forward, en metros, en HEAD coordinates.
    # Esto es lo correcto para usar con mne.head_to_mni.
    rr_head_m = fwd["source_rr"]

    coords_mni305_mm = mne.head_to_mni(
        pos=rr_head_m,
        subject=mri_subject,
        mri_head_t=trans,
        subjects_dir=subjects_dir,
    )

    # Convertimos MNI305 -> MNI152.
    mni305_to_mni152 = np.array([
        [ 0.9975, -0.0073,  0.0176, -0.0429],
        [ 0.0146,  1.0009, -0.0024,  1.5496],
        [-0.0130, -0.0093,  0.9971,  1.1840],
        [ 0.0000,  0.0000,  0.0000,  1.0000],
    ])

    coords_mni_mm = apply_trans(mni305_to_mni152, coords_mni305_mm)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.save(out_file, coords_mni_mm)

    cprint(
        f"MNI152 coords rango {subject_code}: "
        f"x=[{coords_mni_mm[:, 0].min():.1f},{coords_mni_mm[:, 0].max():.1f}], "
        f"y=[{coords_mni_mm[:, 1].min():.1f},{coords_mni_mm[:, 1].max():.1f}], "
        f"z=[{coords_mni_mm[:, 2].min():.1f},{coords_mni_mm[:, 2].max():.1f}] mm"
    )

    return coords_mni_mm






# ============================================================
# Dynemo Parcellation
# ============================================================

def load_parcellation_4d(parcellation_file):
    parc_img = nib.load(parcellation_file)
    parc_data = parc_img.get_fdata()

    if parc_data.ndim == 3:
        labels = np.unique(parc_data)
        labels = labels[labels > 0]

        parc_4d = np.zeros(parc_data.shape + (len(labels),), dtype=float)
        for i, label in enumerate(labels):
            parc_4d[..., i] = (parc_data == label).astype(float)

        parc_img = nib.Nifti1Image(parc_4d, parc_img.affine, parc_img.header)
        parc_data = parc_4d

    if parc_data.ndim != 4:
        raise ValueError(f"La parcellation debe ser 3D o 4D. Shape: {parc_data.shape}")

    return parc_img, parc_data


def get_gridstep_mm(voxel_coords):
    """
    voxel_coords: shape (n_voxels, 3), en mm.
    Estima el gridstep con nearest neighbours.
    """
    coords = np.asarray(voxel_coords, dtype=float)

    tree = scipy.spatial.KDTree(coords)
    dists, _ = tree.query(coords, k=2)

    nn_dists = dists[:, 1]
    nn_dists = nn_dists[np.isfinite(nn_dists)]
    nn_dists = nn_dists[nn_dists > 0]

    if len(nn_dists) == 0:
        raise ValueError("No pude estimar gridstep desde voxel_coords.")

    gridstep = float(np.median(nn_dists))

    # Para el análisis debería ser aproximadamente 8 mm.
    gridstep = float(np.round(gridstep))

    if gridstep <= 0:
        raise ValueError(f"Gridstep inválido: {gridstep}")

    return gridstep


def parcellation_to_voxel_matrix(parcellation_file, voxel_coords):
    """
    Reemplaza la lógica de _resample_parcellation de osl-dynamics.

    Devuelve:
        parcellation_asmatrix: shape (n_voxels, n_parcels)

    voxel_coords debe estar en MNI mm y en el mismo espacio que el atlas.
    """
    parc_img, parc_data = load_parcellation_4d(parcellation_file)

    n_parcels = parc_data.shape[3]
    n_voxels = voxel_coords.shape[0]
    gridstep = get_gridstep_mm(voxel_coords)

    print(f"gridstep = {gridstep} mm")
    print(f"Finding nearest neighbour voxel")

    parcellation_asmatrix = np.zeros((n_voxels, n_parcels), dtype=float)

    for p in range(n_parcels):
        parcel_vol = parc_data[..., p]
        ijk = np.array(np.nonzero(parcel_vol)).T

        if ijk.shape[0] == 0:
            print(f"WARNING: parcela {p} vacía en el atlas.")
            continue

        coords_mm = nib.affines.apply_affine(parc_img.affine, ijk)
        vals = parcel_vol[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

        kdtree = scipy.spatial.KDTree(coords_mm)

        distances, indices = kdtree.query(voxel_coords)

        inside = distances < gridstep
        parcellation_asmatrix[inside, p] = vals[indices[inside]]

    return parcellation_asmatrix


def get_parcel_data_pca(voxel_data, parcellation_asmatrix, method="spatial_basis"):
    """
    Copia funcional de _get_parcel_data_pca de osl-dynamics.

    voxel_data:
        shape (n_voxels, n_time) o (n_voxels, n_time, n_trials)

    parcellation_asmatrix:
        shape (n_voxels, n_parcels)

    method:
        "spatial_basis" o "pca"
    """
    print(f"Calculating parcel time courses with {method}")

    if method not in ["spatial_basis", "pca"]:
        raise ValueError("method debe ser 'spatial_basis' o 'pca'.")

    if parcellation_asmatrix.shape[0] != voxel_data.shape[0]:
        raise ValueError(
            f"Parcellation tiene {parcellation_asmatrix.shape[0]} voxels, "
            f"pero voxel_data tiene {voxel_data.shape[0]}"
        )

    if voxel_data.ndim == 2:
        voxel_data = np.expand_dims(voxel_data, axis=2)
        added_dim = True
    else:
        added_dim = False

    n_parcels = parcellation_asmatrix.shape[1]
    n_time = voxel_data.shape[1]
    n_trials = voxel_data.shape[2]

    voxel_data_reshaped = np.reshape(
        voxel_data,
        (voxel_data.shape[0], n_time * n_trials),
    )

    parcel_data_reshaped = np.zeros((n_parcels, n_time * n_trials))
    voxel_weightings = np.zeros(parcellation_asmatrix.shape)

    temporal_std = np.maximum(
        np.std(voxel_data_reshaped, axis=1),
        np.finfo(float).eps,
    )

    if method == "spatial_basis":

        for pp in range(n_parcels):
            if np.max(np.abs(parcellation_asmatrix[:, pp])) == 0:
                print(f"WARNING: parcela {pp} vacía.")
                continue

            thresh = np.percentile(np.abs(parcellation_asmatrix[:, pp]), 95)

            top_vals = parcellation_asmatrix[
                parcellation_asmatrix[:, pp] > thresh,
                pp,
            ]

            mapsign = np.sign(np.mean(top_vals))
            if mapsign == 0:
                mapsign = 1.0

            scaled_parcellation = (
                mapsign
                * parcellation_asmatrix[:, pp]
                / np.max(np.abs(parcellation_asmatrix[:, pp]))
            )

            positive = scaled_parcellation > 0

            if not np.any(positive):
                print(f"WARNING: parcela {pp} sin voxels positivos.")
                continue

            weighted_ts = voxel_data_reshaped[positive, :]
            weighted_ts = weighted_ts * scaled_parcellation[positive, None]
            weighted_ts = weighted_ts - np.mean(weighted_ts, axis=1, keepdims=True)

            d, U = scipy.sparse.linalg.eigs(weighted_ts @ weighted_ts.T, k=1)
            U = np.real(U)
            d = np.real(d)

            S = np.sqrt(np.abs(d))
            V = weighted_ts.T @ U / S
            pca_scores = S @ V.T

            this_mask = scaled_parcellation[positive] > 0.5

            if np.any(this_mask):
                relative_weighting = np.abs(U[this_mask]) / np.sum(np.abs(U[this_mask]))
                ts_sign = np.sign(np.mean(U[this_mask]))
                if ts_sign == 0:
                    ts_sign = 1.0

                ts_scale = np.dot(
                    relative_weighting.reshape(-1),
                    temporal_std[positive][this_mask],
                )

                node_ts = (
                    ts_sign
                    * ts_scale
                    / np.maximum(np.std(pca_scores), np.finfo(float).eps)
                    * pca_scores
                )

                inds = np.where(positive)[0]

                voxel_weightings[inds, pp] = (
                    ts_sign
                    * ts_scale
                    / np.maximum(np.std(pca_scores), np.finfo(float).eps)
                    * (U.reshape(-1) * scaled_parcellation[positive])
                )

            else:
                print(f"WARNING: máscara vacía para parcela {pp}.")
                node_ts = np.zeros(n_time * n_trials)

            parcel_data_reshaped[pp, :] = node_ts

    elif method == "pca":

        for pp in range(n_parcels):
            mask = parcellation_asmatrix[:, pp] > 0

            if not np.any(mask):
                print(f"WARNING: parcela {pp} vacía.")
                continue

            parcel_ts = voxel_data_reshaped[mask, :]
            parcel_ts = parcel_ts - np.mean(parcel_ts, axis=1, keepdims=True)

            d, U = scipy.sparse.linalg.eigs(parcel_ts @ parcel_ts.T, k=1)
            U = np.real(U)
            d = np.real(d)

            S = np.sqrt(np.abs(d))
            V = parcel_ts.T @ U / S
            pca_scores = S @ V.T

            relative_weighting = np.abs(U) / np.sum(np.abs(U))
            ts_sign = np.sign(np.mean(U))
            if ts_sign == 0:
                ts_sign = 1.0

            ts_scale = np.dot(
                relative_weighting.reshape(-1),
                temporal_std[mask],
            )

            node_ts = (
                ts_sign
                * ts_scale
                / np.maximum(np.std(pca_scores), np.finfo(float).eps)
                * pca_scores
            )

            inds = np.where(mask)[0]
            voxel_weightings[inds, pp] = (
                ts_sign
                * ts_scale
                / np.maximum(np.std(pca_scores), np.finfo(float).eps)
                * U.reshape(-1)
            )

            parcel_data_reshaped[pp, :] = node_ts

    parcel_data = np.reshape(parcel_data_reshaped, (n_parcels, n_time, n_trials))

    if added_dim:
        parcel_data = np.squeeze(parcel_data, axis=2)

    return parcel_data, voxel_weightings


def symmetric_orthogonalisation(timeseries, maintain_magnitudes=True):
    """
    Copia funcional de _symmetric_orthogonalisation de osl-dynamics.

    timeseries:
        shape (n_parcels, n_time) o (n_parcels, n_time, n_trials)
    """
    print("Performing symmetric orthogonalisation")

    if timeseries.ndim == 2:
        timeseries = np.expand_dims(timeseries, axis=2)
        added_dim = True
    else:
        added_dim = False

    nparcels = timeseries.shape[0]
    ntpts = timeseries.shape[1]
    ntrials = timeseries.shape[2]

    A = np.transpose(np.reshape(timeseries, (nparcels, ntpts * ntrials)))

    if maintain_magnitudes:
        D = np.diag(np.sqrt(np.diag(A.T @ A)))
        A = A @ D

    U, S, V = np.linalg.svd(A, full_matrices=False)

    tol = max(A.shape) * S[0] * np.finfo(type(A[0, 0])).eps
    rank = np.sum(S > tol)

    if rank < A.shape[1]:
        raise ValueError(
            f"Not full rank. Rank requerido: {A.shape[1]}, rank actual: {rank}"
        )

    ortho = U @ np.conjugate(V)

    if maintain_magnitudes:
        ortho = ortho @ D

    ortho = np.reshape(
        ortho.T,
        (nparcels, ntpts, ntrials),
    )

    if added_dim:
        ortho = np.squeeze(ortho, axis=2)

    return ortho


def parcellate_spatial_basis_symmetric(
    voxel_data,
    voxel_coords_mni,
    parcellation_file,
):
    """
    Equivalente a:

    parcellation.parcellate(
        fns,
        voxel_data,
        voxel_coords,
        method="spatial_basis",
        orthogonalisation="symmetric",
        parcellation_file=parcellation_file,
    )

    voxel_data:
        shape (n_voxels, n_time)

    voxel_coords_mni:
        shape (n_voxels, 3), en MNI mm.
    """
    print("-----------------")
    print("Parcellating data")
    print("-----------------")

    parcellation_asmatrix = parcellation_to_voxel_matrix(
        parcellation_file=parcellation_file,
        voxel_coords=voxel_coords_mni,
    )

    parcel_data, voxel_weightings = get_parcel_data_pca(
        voxel_data=voxel_data,
        parcellation_asmatrix=parcellation_asmatrix,
        method="spatial_basis",
    )

    parcel_data = symmetric_orthogonalisation(
        parcel_data,
        maintain_magnitudes=True,
    )

    return parcel_data, voxel_weightings, parcellation_asmatrix


def convert_to_mne_raw(parcel_data, raw, ch_names=None, extra_chans="stim"):
    """
    Equivalente simplificado a convert_to_mne_raw de osl-dynamics.
    """
    if isinstance(extra_chans, str):
        extra_chans = [extra_chans]
    elif extra_chans is None:
        extra_chans = []

    if raw.get_data().shape[1] != parcel_data.shape[1]:
        _, times = raw.get_data(reject_by_annotation="omit", return_times=True)
        indices = raw.time_as_index(times, use_rounding=True)
        indices = indices[: parcel_data.shape[1]]

        full_data = np.zeros((parcel_data.shape[0], len(raw.times)), dtype=np.float32)
        full_data[:, indices] = parcel_data
    else:
        full_data = parcel_data

    if ch_names is None:
        ch_names = [f"parcel_{i}" for i in range(full_data.shape[0])]

    info = mne.create_info(
        ch_names=ch_names,
        ch_types="misc",
        sfreq=raw.info["sfreq"],
    )

    parc_raw = mne.io.RawArray(full_data, info)

    with parc_raw.info._unlock():
        parc_raw.info["highpass"] = float(raw.info["highpass"])
        parc_raw.info["lowpass"] = float(raw.info["lowpass"])

    parc_raw.set_meas_date(raw.info["meas_date"])
    parc_raw.set_annotations(raw.annotations)

    for extra_chan in extra_chans:
        try:
            chan_raw = raw.copy().pick(extra_chan)
            parc_raw.add_channels([chan_raw], force_update_info=True)
        except Exception:
            pass

    parc_raw.info["description"] = raw.info.get("description", None)

    return parc_raw


def save_parcel_data_as_fif(parcel_data, raw, filename, extra_chans="stim"):
    """
    Equivalente a parcellation.save_as_fif(...).
    """
    print(f"Saving {filename}")

    parc_raw = convert_to_mne_raw(
        parcel_data=parcel_data,
        raw=raw,
        ch_names=[f"parcel_{i}" for i in range(parcel_data.shape[0])],
        extra_chans=extra_chans,
    )

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    parc_raw.save(filename, overwrite=True)

    return parc_raw

