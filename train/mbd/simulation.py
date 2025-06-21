from .interactiveplot import interactive_legend
from .observables import *
from .timeseries import *
from .trajectories import *

import logging
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import operator
from functools import reduce


class Stage:

    def __init__(self, simulation, stageStats):
        self._simulation = simulation
        self.stageId = stageStats["stage"]
        self.stepsMade = stageStats["stepsMade"]
        self.tBefore = stageStats["tBefore"]
        self.tAfter = stageStats["tAfter"]
        self.tDiff = self.tAfter - self.tBefore
        for q in ["Timeseries", "Trajectories", "Scalar", "Onebody", "Radial", "TwobodyPlanar", "Clusters"]:
            try:
                setattr(self, f"_{q.lower()}", globals()[q](self._simulation.path, self))
            except FileNotFoundError:
                pass

    def getConfig(self, keys):
        try:
            return reduce(operator.getitem, keys, self._simulation.config["Stages"][self.stageId])
        except KeyError:
            return reduce(operator.getitem, keys, self._simulation.config)

    @property
    def timeseries(self) -> Timeseries:
        try:
            return self._timeseries
        except AttributeError:
            raise AttributeError(f"No timeseries found for stage {self.stageId}")

    @property
    def trajectories(self) -> Trajectories:
        try:
            return self._trajectories
        except AttributeError:
            raise AttributeError(f"No trajectories found for stage {self.stageId}")

    @property
    def scalar(self) -> Scalar:
        try:
            return self._scalar
        except AttributeError:
            raise AttributeError(f"No scalars found for stage {self.stageId}")

    @property
    def onebody(self) -> Onebody:
        try:
            return self._onebody
        except AttributeError:
            raise AttributeError(f"No onebody fields found for stage {self.stageId}")

    @property
    def radial(self) -> Radial:
        try:
            return self._radial
        except AttributeError:
            raise AttributeError(f"No radial distribution functions found for stage {self.stageId}")

    @property
    def twobodyplanar(self) -> TwobodyPlanar:
        try:
            return self._twobodyplanar
        except AttributeError:
            raise AttributeError(f"No twobody planar distribution functions found for stage {self.stageId}")

    @property
    def clusters(self) -> Clusters:
        try:
            return self._clusters
        except AttributeError:
            raise AttributeError(f"No clusters found for stage {self.stageId}")


class StageCollection:

    def __init__(self, stages: list[list[Stage]]):
        self.stages = stages

    def __getitem__(self, stageId: Union[tuple[int, int], int]) -> Stage:
        run = 0
        if type(stageId) is tuple:
            run, stage = stageId
        else:
            stage = stageId
        try:
            self.stages[run]
        except IndexError:
            raise IndexError(f"There is no run {run}")
        try:
            self.stages[run][stage]
        except IndexError:
            raise IndexError(f"There is no stage {stage}")
        if self.stages[run][stage] is None:
            raise IndexError(f"Stage {stage} has not been parsed")
        return self.stages[run][stage]

    @property
    def average(self) -> Stage:
        try:
            return self._average
        except AttributeError:
            stagesFlat = [s for run in self.stages for s in run]
            for s in stagesFlat:
                if s is not None:
                    break
            self._average = copy.deepcopy(s)
            for q in ["Onebody", "Radial", "TwobodyPlanar"]:
                try:
                    averageQuantity = getattr(self._average, q.lower())
                    for species in averageQuantity._data.keys():
                        for key in averageQuantity._data[species].keys():
                            dataArray = [getattr(s, q.lower())[key, species] for s in stagesFlat if s is not None]
                            weightArray = [s.tDiff for s in stagesFlat if s is not None]
                            averageQuantity._data[species][key] = np.average(dataArray, weights=weightArray, axis=0)
                except AttributeError:
                    pass
            return self._average

    def plotOnebodySuccessive(self, key: Union[tuple[str,...], str], species: Union[tuple[int,...], int] = 0, dim: int = 1, interactive: bool = False, **kwargs):
        """
        Plot the specified onebody fields successively, i.e. show a figure for each stage.
        """
        if dim == 2 and (type(key) is tuple or type(species) is tuple):
            raise ValueError("Field or species cannot be tuple in 2D plotting")
        if dim != 2 and interactive:
            raise ValueError("Interactive is only possible when dim=2")
        for irun, run in enumerate(self.stages):
            for istage, s in enumerate(run):
                if s is not None:
                    try:
                        s.onebody
                    except AttributeError:
                        continue
                    suptitle = f"Run {irun}, Stage {istage}"
                    if dim == 2:
                        if interactive:
                            s.onebody.plot2DInteractive(key, species, **kwargs)
                        else:
                            s.onebody.plot2D(key, species, suptitle=suptitle, **kwargs)
                    if dim == 1:
                        s.onebody.plot1D(key, species, suptitle=suptitle, **kwargs)

    def plotOnebodyAll(self, key: Union[tuple[str,...], str], species: Union[tuple[int,...], int] = 0, average: Union[bool, Literal["only"]] = False, **kwargs):
        """
        Plot the specified onebody fields of all stages in one figure.
        """
        fig, ax = plt.subplots()

        for irun, run in enumerate(self.stages):
            for istage, s in enumerate(run):
                if s is not None:
                    try:
                        s.onebody
                    except AttributeError:
                        continue
                    if average != "only":
                        s.onebody.plot1D(key, species, fig=fig, ax=ax, addToLabel=f"Run {irun}, Stage {istage}", **kwargs)
        if average:
            self.average.onebody.plot1D(key, species, fig=fig, ax=ax, addToLabel=f"average", linewidth=2, color=None if average == "only" else "black", **kwargs)
        interactive_legend(ax).show()

    def plotTimeseriesAll(self, quantity: str, mode: Literal["steps", "time"]="time", log=False, **kwargs):
        fig, ax = plt.subplots()
        for irun, run in enumerate(self.stages):
            for istage, s in enumerate(run):
                if s is not None:
                    try:
                        s.timeseries
                    except AttributeError:
                        continue
                    s.timeseries.plot(quantity, mode=mode, log=log, fig=fig, ax=ax, label=f"{quantity}, Run {irun}, Stage {istage}", **kwargs)
        plt.show()


class Simulation(StageCollection):

    def __init__(self, path, stageIds=None):
        logging.info(f"STARTING SIMULATION ANALYSIS OF {path}")
        self.path = path
        with open(self.path + "/config.yaml", 'r') as c:
            self.config = yaml.safe_load(c)
        self._parseStats()
        self._parseStages(stageIds)

    def _parseStats(self, filename="stats.dat"):
        self.stats: np.ndarray = np.genfromtxt(self.path + "/" + filename, dtype=None, names=True)
        self.stats = np.atleast_1d(self.stats)
        logging.info(f"We have {len(self.stats)} stages:")
        logging.info(pd.DataFrame(self.stats))

    def _parseStages(self, stageIds):
        numStages = len(self.stats)
        stages = [None for _ in range(numStages)]
        if stageIds is None:
            stageIds = list(range(numStages))
        for stageId in stageIds:
            logging.info(f"- Reading stage {stageId} -")
            if stageId < len(self.stats):
                stages[stageId] = Stage(self, self.stats[stageId])
            else:
                logging.warning(f"Warning: Stage {stageId} not available!")
        super().__init__([stages])
        logging.info(f"\nSimulation {self.path} complete!\n")


class Ensemble(StageCollection):

    def __init__(self, path, runIds=None, stageIds=None):
        from os.path import exists
        self.path = path
        logging.info(f"STARTING ENSEMBLE ANALYSIS OF {self.path}")
        if exists(self.path + "/stats.dat") and runIds is None:
            self.runDirs = [self.path]
        else:
            if runIds is None:
                import glob
                self.runDirs = glob.glob(self.path + "/run*")
            else:
                self.runDirs = [self.path + "/run" + str(runId) for runId in runIds]
        if len(self.runDirs) == 0:
            raise FileNotFoundError("Could not find ensemble data")
        self.config = Simulation(self.runDirs[0], []).config
        stages = []
        for d in self.runDirs:
            run = Simulation(d, stageIds=stageIds)
            stages.append(run.stages[0])
        super().__init__(stages)
        logging.info(f"Ensemble {self.path} done!\n")
