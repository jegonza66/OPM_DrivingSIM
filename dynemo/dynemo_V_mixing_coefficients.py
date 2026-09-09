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
import setup
import load
import functions_analysis
import matplotlib.pyplot as plt
from general_utility_functions import cprint, rprint, yprint, gprint
import dynemo__mixing_coefficients_utils as mc

# Setup
exp_info = setup.exp_info()
subjects = exp_info.subjects_ids

# Run parameters (must match the trained model in dynemo_II / dynemo_III)
from dynemo_config import n_modes, n_pca, n_embeddings, sequence_length, ch_picks

# Paths:
dynemo_infered_parameters_path = paths.dynemo_run_save_path(n_modes, n_pca, n_embeddings, sequence_length, ch_picks, "DyNeMo_Infered_Parameters")
dynemo_plots_mixing_coefficients_path = paths.dynemo_run_plots_path(n_modes, n_pca, n_embeddings, sequence_length, ch_picks, "Mixing_Coefficients")

FS = mc.FS_ALPHA
N_FIRST = 2000          # samples of the "first seconds" plot (8 s at 250 Hz)
FULL_WIN = FS           # full-series downsampling: mean over 1 s windows
EYEMAP_PRE = 10.0       # seconds before the first eyemap trigger
EYEMAP_FIRST = "em_blink"
EYEMAP_END = "inst3"    # onset of the next instruction marks the end of the eyemap block


def save_pair(alpha, times, out_dir, subject_code, tag, events=None, heat_alpha=None):
    """Save stack + heatmap versions of one alpha window."""
    title = f"{subject_code} - {tag}"
    fig = mc.plot_alpha_stack(alpha, times, title=title, events=events)
    mc.save_figure(fig, os.path.join(out_dir, f"{subject_code}_{tag}_stack.png"))
    plt.close(fig)
    fig = mc.plot_alpha_heatmap(alpha if heat_alpha is None else heat_alpha, times,
                                title=title, events=events)
    mc.save_figure(fig, os.path.join(out_dir, f"{subject_code}_{tag}_heatmap.png"))
    plt.close(fig)


for use_reweighted in (False, True):
    alpha_all = mc.load_alpha(use_reweighted=use_reweighted,
                              infered_parameters_path=dynemo_infered_parameters_path)
    out_dir = os.path.join(dynemo_plots_mixing_coefficients_path,
                           "alpha_reweighted" if use_reweighted else "alpha")
    os.makedirs(out_dir, exist_ok=True)

    for i, subject_code in enumerate(subjects):
        if i >= len(alpha_all):
            rprint(f">>> No hay alpha para {subject_code}")
            continue
        alpha = alpha_all[i]
        cprint(f">>> {subject_code} ({'reweighted' if use_reweighted else 'raw'} alpha)")

        ####### FIRST SECONDS #######
        a = alpha[:N_FIRST]
        save_pair(a, np.arange(len(a)) / FS, out_dir, subject_code, "first8s")

        ####### RECORDING TIMELINE (continuous 250 Hz, zeros in gaps) #######
        mode_raw, valid_mask, times = mc.build_mode_raw(
            subject_code=subject_code, alpha_i=alpha, ch_picks=ch_picks,
            n_pca=n_pca, n_embeddings=n_embeddings, sequence_length=sequence_length)
        meg_data = load.meg(subject_id=subject_code, meg_params={"data_type": "processed"})
        annot = meg_data.annotations
        desc = np.asarray(annot.description)
        mode_data = mode_raw.get_data()

        ####### FULL SERIES (1 s means on the recording timeline) #######
        n_win = mode_data.shape[1] // FULL_WIN
        a = mode_data[:, :n_win * FULL_WIN].reshape(-1, n_win, FULL_WIN).mean(axis=2).T
        a_heat = a.copy()
        a_heat[valid_mask[:n_win * FULL_WIN].reshape(n_win, FULL_WIN).mean(axis=1) < 0.5] = np.nan
        phases = functions_analysis.get_experiment_phase_times(subject_code, meg_data)
        phase_marks = {"CF onset": phases["CF"][0], "DA start": phases["DA"][0],
                       "Audio start": phases["Audio"][0], "CF end": phases["CF"][1]}
        save_pair(a, np.arange(n_win) + 0.5, out_dir, subject_code, "full",
                  events=phase_marks, heat_alpha=a_heat)

        ####### EYEMAP WINDOW #######
        # 0-based seconds, same convention as mc.get_event_onsets_seconds
        em_events = {d: annot.onset[desc == d][0] - meg_data.first_time
                     for d in np.unique(desc) if d.startswith("em_")}
        if EYEMAP_FIRST not in em_events or EYEMAP_END not in desc:
            yprint(f">>> {subject_code}: sin '{EYEMAP_FIRST}' / '{EYEMAP_END}', salto eyemap")
            continue
        t_start = em_events[EYEMAP_FIRST] - EYEMAP_PRE
        t_end = annot.onset[desc == EYEMAP_END][0] - meg_data.first_time
        sel = (times >= t_start) & (times <= t_end)

        a = mode_data[:, sel].T
        a_heat = a.copy()
        a_heat[valid_mask[sel] < 0.5] = np.nan
        rel_times = times[sel] - em_events[EYEMAP_FIRST]
        rel_events = {d: t - em_events[EYEMAP_FIRST] for d, t in em_events.items()}
        save_pair(a, rel_times, out_dir, subject_code, "eyemap",
                  events=rel_events, heat_alpha=a_heat)

gprint(f">>> Plots guardados en {dynemo_plots_mixing_coefficients_path}")