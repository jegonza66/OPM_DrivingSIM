"""Central DyNeMo run parameters (single source of truth).

These four values define a DyNeMo "run" and are baked into the per-run output
folder name (e.g. ``modes6_emb15_seq100``). Every ``dynemo_*`` script imports
them from here, so a run is configured in exactly one place and the saving /
plotting paths can never drift between steps.

This file is COMMITTED and should keep the project default values below.

To use your OWN run parameters WITHOUT committing the change, create a file
named ``dynemo_config_local.py`` next to this one and (re)define any of the
names below. That file is gitignored, so it is never tracked. Example:

    # dynemo_config_local.py  (untracked)
    n_modes = 8
    sequence_length = 200

Anything not redefined there falls back to the committed defaults here.
"""

# --- Committed defaults --------------------------------------------------
n_modes = 6           # number of DyNeMo modes
n_pca = 80            # number of TDE-PCA components (n_channels for the model)
n_embeddings = 15     # number of time-delay embeddings
sequence_length = 100  # model sequence length

# --- Optional untracked local override -----------------------------------
# If a sibling ``dynemo_config_local.py`` exists, any of the four names it
# defines replaces the corresponding default above.
try:
    import dynemo_config_local as _local
except ImportError:
    _local = None

if _local is not None:
    for _name in ("n_modes", "n_pca", "n_embeddings", "sequence_length"):
        if hasattr(_local, _name):
            globals()[_name] = getattr(_local, _name)
    print(f"[dynemo_config] Using local override: "
          f"n_modes={n_modes}, n_pca={n_pca}, "
          f"n_embeddings={n_embeddings}, sequence_length={sequence_length}")

