"""Training helpers shared by the DyNeMo model stage."""

import os
import pickle

import numpy as np
import tensorflow as tf


class RestoreBestWeightsAfter(tf.keras.callbacks.Callback):
    """Restore the lowest-loss weights observed from a given epoch onward."""

    def __init__(self, start_epoch):
        super().__init__()
        self.start_epoch = int(start_epoch)
        self.best_loss = np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        loss = (logs or {}).get("loss")
        if epoch < self.start_epoch or loss is None or not np.isfinite(loss):
            return
        if loss < self.best_loss:
            self.best_loss = float(loss)
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


class TrainingCheckpoint(tf.keras.callbacks.Callback):
    """Save resumable training state every ``every`` epochs.

    Stores model and optimizer variables, the completed epoch, per-epoch losses
    and the state of ``best_weights`` so an interrupted ``fit`` can resume
    with ``initial_epoch``.
    """

    def __init__(self, osl_model, directory, every, best_weights):
        super().__init__()
        self.osl_model = osl_model
        self.directory = directory
        self.every = int(every)
        self.best_weights = best_weights
        self.losses = []
        self._state_file = os.path.join(directory, "state.pkl")

    def save(self, epoch):
        """Save state after ``epoch`` completed epochs."""
        # get_weights/optimizer.variables rather than tf.train.Checkpoint, which
        # misses Keras 3 weights nested inside the RNN wrapper layers.
        keras_model = self.osl_model.model
        if not keras_model.optimizer.built:
            keras_model.optimizer.build(keras_model.trainable_variables)
        state = {
            "epoch": epoch,
            "model": keras_model.get_weights(),
            "optimizer": [v.numpy() for v in keras_model.optimizer.variables],
            "losses": self.losses[:epoch],
            "best_loss": self.best_weights.best_loss,
            "best_weights": self.best_weights.best_weights,
        }
        os.makedirs(self.directory, exist_ok=True)
        # Write then rename, so a crash mid-save keeps the previous valid state.
        tmp_file = self._state_file + ".tmp"
        with open(tmp_file, "wb") as f:
            pickle.dump(state, f)
        os.replace(tmp_file, self._state_file)

    def restore(self):
        """Restore the latest state; return the completed epoch or None."""
        if not os.path.exists(self._state_file):
            return None
        with open(self._state_file, "rb") as f:
            state = pickle.load(f)
        keras_model = self.osl_model.model
        optimizer = keras_model.optimizer
        initial_lr = float(optimizer.learning_rate.numpy())
        keras_model.set_weights(state["model"])
        if not optimizer.built:
            optimizer.build(keras_model.trainable_variables)
        for variable, value in zip(optimizer.variables, state["optimizer"], strict=True):
            variable.assign(value)
        # osl-dynamics derives its learning-rate schedule from the optimizer's current value.
        optimizer.learning_rate.assign(initial_lr)
        self.losses = list(state["losses"])
        self.best_weights.best_loss = state["best_loss"]
        self.best_weights.best_weights = state["best_weights"]
        return state["epoch"]

    def on_epoch_begin(self, epoch, logs=None):
        # Saving here (not on_epoch_end) captures the KL factor osl-dynamics updates after user callbacks.
        if epoch and epoch % self.every == 0:
            self.save(epoch)

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(float((logs or {}).get("loss", np.nan)))