"""Small synthetic checks for the DyNeMo stats / plotting helpers.

Run:  python dynemo_self_check.py
"""

import sys
import os

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = r"D:\OneDrive - The University of Nottingham\OPM-MEG-analysis - OPM2\Scripts\dynemo"

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dynemo__mixing_coefficients_utils as mc
import dynemo_VIII_phase_entropy as phase


def main():
    rng = np.random.default_rng(3)

    # temporal_cluster_test: TFCE intervals cover the injected effect, nothing else
    data = rng.normal(size=(10, 40)) * 0.5
    data[:, 15:25] += 2.0
    mask, found = mc.temporal_cluster_test(
        data, t_thresh={"start": 0.0, "step": 0.2}, n_permutations=256,
        return_clusters=True)
    assert mask[15:25].all() and not mask[:10].any()
    assert all(15 <= start < stop <= 30 for start, stop, _ in found)
    assert all(p < 0.05 for _, _, p in found)
    # fixed-threshold path returns the same shape
    mask_t, found_t = mc.temporal_cluster_test(
        data, t_thresh=0.05, n_permutations=64, return_clusters=True)
    assert mask_t.shape == (40,) and found_t

    # stars
    assert mc.stars(0.0005) == "***" and mc.stars(0.2) == "n.s."

    # draw_significance_bars grows the figure and extends the y axis
    fig, ax = plt.subplots(figsize=(6, 4))
    times = np.linspace(-1, 1, 40)
    ax.plot(times, np.zeros(40))
    height = fig.get_figheight()
    top = ax.get_ylim()[1]
    mc.draw_significance_bars(ax, times, [found, []], ["tab:blue", "tab:red"])
    assert fig.get_figheight() > height and ax.get_ylim()[1] > top
    plt.close(fig)

    # entropy: pure profile 0 bits, uniform over 4 modes 2 bits, NaN propagates
    assert np.isclose(phase._entropy(np.array([1.0, 0, 0, 0])), 0.0)
    assert np.isclose(phase._entropy(np.full(4, 0.25)), 2.0)
    assert np.isnan(phase._entropy(np.array([np.nan, 0.5, 0.5])))

    # holm: monotone, bounded, smallest p multiplied by m
    adjusted = phase._holm([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])

    # JSD: identical subjects -> 0; each subject in its own mode -> log2(n)
    same = np.tile(np.array([0.7, 0.2, 0.1, 0.0]), (8, 3, 1))
    assert np.allclose(phase._jsd_per_window(same), 0.0)
    apart = np.stack([np.tile(np.eye(4)[s % 4], (3, 1)) for s in range(8)])
    assert np.allclose(phase._jsd_per_window(apart), 2.0)
    # too few subjects -> NaN
    assert np.isnan(phase._jsd_per_window(same[:3])).all()

    print("dynemo self-check OK")


if __name__ == "__main__":
    main()
