from typing import Literal

import logging
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt


class Timeseries:

    def __init__(self, path, stage):
        fromFile = self._parse(path, stage.stageId)
        logging.info(f"Got timeseries from file {fromFile}")

    def __getitem__(self, key: str) -> np.ndarray:
        try:
            return self.data[key]
        except KeyError:
            raise KeyError(f"There is no quantity '{key}' in this timeseries")

    def _parseHDF5(self, filename):
        self.time = np.array([]) # time is always the same
        self.data = {}
        with h5py.File(filename, "r", swmr=True, locking=False) as f:
            group = f["observables"]["timeseries"]
            self.time = np.array(group["time"])
            keys = list(group.keys())
            for k in keys:
                if k == "time":
                    continue
                self.data[k] = np.array(group[k]["value"])

    def _parseASCII(self, filename):
        df = pd.read_table(filename, sep=" ")
        self.data = {i: df[i].to_numpy() for i in df.columns}

    def _parse(self, path, stageId):
        try:  # Data as HDF5
            filename = f"{path}/{stageId}.h5md"
            self._parseHDF5(filename)
            return filename
        except (OSError, KeyError):
            pass
        try:  # Data as ASCII
            filename = f"{path}/{stageId}_timeseries.dat"
            self._parseASCII(filename)
            return filename
        except OSError:
            pass
        raise FileNotFoundError(f"Could not parse timeseries data.")

    def plot(self, quantity: str, mode: Literal["steps", "time"]="time", log=False, fig=None, ax=None, **kwargs):
        showAtEnd = False
        if fig is None and ax is None:
            fig, ax = plt.subplots()
            showAtEnd = True
        if log:
            ax.set_yscale("log")
        if mode == "steps":
            ax.plot(self[quantity], **kwargs)
        elif mode == "time":
            ax.plot(self.time, self[quantity], **kwargs)
        else:
            raise ValueError("Unknown timeseries plotting mode. Choose 'steps' or 'time'.")
        ax.legend()
        if showAtEnd:
            plt.show()

    def average(self, quantity: str):
        try:
            self["weights"]
            return np.average(self[quantity], weights=self["weights"])
        except KeyError:
            from scipy import integrate
            return integrate.trapezoid(self[quantity], self.time)
