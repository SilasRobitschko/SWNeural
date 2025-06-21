import logging
import numpy as np
import h5py


class TrajectoryFrame:

    def __init__(self, filename, frame):
        self._parse(filename, frame)

    def __getitem__(self, key: str) -> np.ndarray:
        try:
            return self.data[key]
        except KeyError:
            raise KeyError(f"There is no quantity '{key}' in this trajectory frame")

    def _parse(self, filename, frame):
        self.data = {}
        with h5py.File(filename, "r", swmr=True, locking=False) as f:
            group = f["particles"]
            self.time = group["time"][frame]
            quantities = list(group.keys())
            for q in quantities:
                if q == "time" or q == "box":
                    continue
                self.data[q] = np.array(group[q]["value"][frame])


class Trajectories:

    def __init__(self, path, stage):
        self.path = path
        self.stage = stage
        fromFile = self._parse(path, stage.stageId)
        logging.info(f"Trajectories available in file {fromFile}")

    def __getitem__(self, frame: int) -> TrajectoryFrame:
        if frame >= len(self.times):
            raise IndexError(f"There is no frame {frame} in this trajectory, only {len(self.times)} are available")
        return TrajectoryFrame(f"{self.path}/{self.stage.stageId}.h5md", frame)

    def _parse(self, path, stageId):
        try:
            filename = f"{path}/{stageId}.h5md"
            self._parseHDF5(filename)
            return filename
        except (OSError, KeyError):
            pass
        raise FileNotFoundError(f"Could not parse trajectory data.")

    def _parseHDF5(self, filename):
        with h5py.File(filename, "r", swmr=True, locking=False) as f:
            # Only try to get the group and the time stamps of the frames
            # The actual trajectory data is huge, so load it only frame by frame when needed
            group = f["particles"]
            self.times = np.array(group["time"])
