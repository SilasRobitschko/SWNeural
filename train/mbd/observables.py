from typing import Union

import logging
import re
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.widgets


class Observables:

    def __init__(self, path, stage):
        self.stage = stage
        self._data = {}
        self.axisLabels = []
        self._coordinates = []
        fromFile = self._parse(path, stage.stageId)
        logging.info(f"Got {self.obsType} from file {fromFile}")

    def _normalizeKeyAndMaybeSpecies(self, keyAndMaybeSpecies: Union[tuple[str, tuple[int, ...]], tuple[str, int], str]) -> tuple[str, tuple[int, ...]]:
        '''
        Utility function that is used in the operator methods (__getitem__, ...) below.
        Returns `(key, (species0, species1, ...))` where species indices are filled with default species 0 if not otherwise specified.
        '''
        species = tuple(0 for _ in range(self.speciesIndices))
        if type(keyAndMaybeSpecies) is tuple:
            key, species = keyAndMaybeSpecies
            if type(species) is not tuple:
                species = tuple(species for _ in range(self.speciesIndices))
        else:
            key = keyAndMaybeSpecies
        return key, species

    def _getComputedFunction(self, key):
        '''
        To be implemented by subclasses, should return a lambda function that can be used to calculate an additonal observable `key` from existing ones.
        For unknown `key`, None should be returned.
        '''
        return None

    def _calcComputed(self, key):
        func = self._getComputedFunction(key)
        if not func:
            raise ValueError(f"Unknown key '{key}'")
        try:
            for s in self.species():
                    self._data[s][key] = func(s)
        except KeyError:
            raise KeyError(f"Cannot compute '{key}' because required data is missing")

    def __getitem__(self, keyAndMaybeSpecies: Union[tuple[str, tuple[int, ...]], tuple[str, int], str]) -> np.ndarray:
        key, species = self._normalizeKeyAndMaybeSpecies(keyAndMaybeSpecies)
        try:
            self._data[species]
        except KeyError:
            raise KeyError(f"There is no species {species}")
        try:
            return self._data[species][key]
        except KeyError:
            pass
        try:
            self._calcComputed(key)
            return self._data[species][key]
        except ValueError:
            raise KeyError(f"There is no observable '{key}'")

    def __contains__(self, keyAndMaybeSpecies: Union[tuple[str, tuple[int, ...]], tuple[str, int], str]):
        key, species = self._normalizeKeyAndMaybeSpecies(keyAndMaybeSpecies)
        try:
            self._data[species]
        except KeyError:
            return False
        try:
            self._data[species][key]
            return True
        except KeyError:
            pass
        return self._getComputedFunction(key) is not None

    def species(self):
        return self._data.keys()

    def keys(self):
        species = tuple(0 for _ in range(self.speciesIndices))
        return self._data[species].keys()

    def _parse(self, path, stageId):
        try:  # Data as HDF5
            filename = f"{path}/{stageId}.h5md"
            self._parseHDF5(filename)
            return filename
        except (OSError, KeyError):
            pass
        try:  # Data as ASCII
            filename = f"{path}/{stageId}_{self.obsType}.dat"
            self._parseASCII(filename)
            return filename
        except OSError:
            pass
        raise FileNotFoundError(f"Could not parse {self.obsType} data")

    def _parseHDF5(self, filename):
        with h5py.File(filename, "r", swmr=True, locking=False) as f:
            group = f["observables"][self.obsType]

            if "coordinates" in group:
                coordinates = group["coordinates"]
                for d in coordinates.keys():
                    try:
                        label = coordinates[d].attrs["label"]
                    except KeyError:
                        label = f"coord_{d}"
                    self.axisLabels.append(label)
                    self._coordinates.append(np.array(coordinates[d]))

            species = [int(k) for k in list(group.keys()) if k.isdigit()]
            def iterateSpecies(speciesTuple=()):
                nonlocal species
                if len(speciesTuple) == self.speciesIndices:
                    yield speciesTuple
                else:
                    for s in species:
                        yield from iterateSpecies(speciesTuple + (s,))

            for speciesTuple in iterateSpecies():
                speciesGroup = group
                for s in speciesTuple:
                    speciesGroup = speciesGroup[str(s)]
                self._data[speciesTuple] = {}
                for key in speciesGroup.keys():
                    self._data[speciesTuple][key] = np.array(speciesGroup[key])

    def _parseASCII(self, filename):
        try:
            with open(filename + ".coord") as f:
                for line in f.readlines():
                    label, coords = line.split(':')
                    self.axisLabels.append(label)
                    self._coordinates.append(np.fromstring(coords, sep=' '))
        except FileNotFoundError:
            pass
        shape = tuple(len(cs) - 1 for cs in self._coordinates)
        with open(filename) as f:
            for line in f.readlines():
                keyAndSpecies, data = line.split(':')
                key = keyAndSpecies.split(' ')[0]
                speciesTokens = keyAndSpecies.split(' ')[1:]
                speciesTuple = tuple(int(s) for s in speciesTokens)
                if speciesTuple not in self._data:
                    self._data[speciesTuple] = {}
                self._data[speciesTuple][key] = np.fromstring(data, sep=' ')
                if len(shape) > 1:
                    self._data[speciesTuple][key] = self._data[speciesTuple][key].reshape(shape)


class Fields(Observables):

    def __init__(self, path, stage):
        super().__init__(path, stage)
        self.r = self._coordinates
        self.dim = len(self.r)
        self.shape = list(list(self._data.values())[0].values())[0].shape
        self.dr = [r[1] - r[0] if len(r) > 1 else 0 for r in self.r]
        self.rCenter = [self.r[d][:self.shape[d]] + 0.5 * self.dr[d] for d in range(self.dim)]

    def _real2index(self, real, axis):
        if real is None or axis is None:
            return None
        if real > self.r[axis][-1]:
            raise ValueError(f"coordinate = {real} for axis {axis} ({self.axisLabels[axis]}) out of bounds (max coordinate {self.r[axis][-1]})")
        if self.dr[axis] == 0:
            return 0
        return int(real / self.dr[axis])

    def _getRemainingAxis(self, axis):
        axis = np.atleast_1d(axis)
        return tuple(i for i in range(self.dim) if i not in axis)

    def _getSlice(self, posind, axis):
        posind = np.atleast_1d(posind)
        remainingAxis = self._getRemainingAxis(axis)
        slc = list(slice(None) for _ in range(self.dim))
        for p, a in zip(posind, remainingAxis):
            slc[a] = p
        return tuple(slc)


class Scalar(Fields):

    def __init__(self, path, stage):
        self.obsType = "scalar"
        self.speciesIndices = 0
        super().__init__(path, stage)
        for key in self._data[()].keys():
            self._data[()][key] = self._data[()][key].item()


class Onebody(Fields):

    def __init__(self, path, stage):
        self.obsType = "onebody"
        self.speciesIndices = 1
        super().__init__(path, stage)
        if "coord_0" in self.axisLabels:
            self.axisLabels = ["x", "y", "z"]

    def _getComputedFunction(self, key):
        if key == "rho":
            return lambda s: self["1", s]
        if key == "eext":
            return lambda s: self["Eext", s] / self["1", s]
        if key == "muloc":
            mu = self.stage.getConfig(["System", "mu"])
            return lambda s: mu - self["eext", s]
        if key == "c1":
            mu = self.stage.getConfig(["System", "mu"])
            beta = 1 / self.stage.getConfig(["System", "T"])
            return lambda s: np.log(self["rho", s]) + beta * (self["eext", s] - mu)
        if match := re.fullmatch(r"fint([xyz])", key):
            d = match.group(1)
            return lambda s: self["Fint"+d, s] / self["1", s]
        if match := re.fullmatch(r"fext([xyz])", key):
            d = match.group(1)
            return lambda s: self["Fext"+d, s] / self["1", s]
        if match := re.fullmatch(r"F([xyz])", key):
            d = match.group(1)
            return lambda s: self["Fext"+d, s] + self["Fint"+d, s]
        if match := re.fullmatch(r"f([xyz])", key):
            d = match.group(1)
            return lambda s: self["F"+d, s] / self["1", s]
        if match := re.fullmatch(r"JRand([xyz])", key):
            d = match.group(1)
            return lambda s: self["rRand"+d+"_dt", s] / 2 # See https://doi.org/10.1103/PhysRevE.99.023306 Appendix A2
        if match := re.fullmatch(r"vRand([xyz])", key):
            d = match.group(1)
            return lambda s: self["JRand"+d, s] / self["1", s]
        if match := re.fullmatch(r"J([xyz])", key):
            d = match.group(1)
            gamma = 1 # TODO: generalize, we might need additional particle info for this
            return lambda s: self["F"+d, s] / gamma + self["JRand"+d, s]
        if match := re.fullmatch(r"v([xyz])", key):
            d = match.group(1)
            return lambda s: self["J"+d, s] / self["1", s]
        return None

    def plot1D(self, quantities, species=(0,), pos=None, axis=None, suptitle=None, fig=None, ax=None, addToLabel=None, **kwargs):
        quantities = np.atleast_1d(quantities)
        species = np.atleast_1d(species)
        if pos is None:
            pos = tuple(0 for _ in range(self.dim - 1))
        if axis is None:
            axis = self.dim - 1
        showAtEnd = False
        if fig is None and ax is None:
            fig, ax = plt.subplots()
            showAtEnd = True
        remainingAxis = self._getRemainingAxis(axis)
        posind = tuple(self._real2index(p, a) for p, a in zip(pos, remainingAxis))
        lineSlice = self._getSlice(posind, axis)
        for f in quantities:
            label = f
            for s in species:
                label += "_" + str(s)
                label += ", " + addToLabel if addToLabel else ""
                ax.plot(self.rCenter[axis], self[f, s][lineSlice], label=label, **kwargs)
        ax.set_xlabel(f'${self.axisLabels[axis]}$')
        title = [f'${self.axisLabels[a]} = {self.rCenter[a][pind]}$' for a, pind in zip(remainingAxis, posind)]
        ax.set_title(", ".join(title))
        if suptitle:
            fig.suptitle(suptitle)
        ax.legend()
        if showAtEnd:
            plt.show()

    def _plot2D(self, ax, quantity, species, posind, axis, vmin=None, vmax=None, **kwargs):
        planeSlice = self._getSlice(posind, axis)
        xlim = [self.r[axis[0]][0], self.r[axis[0]][-1]]
        ax.set_xlabel(f"${self.axisLabels[axis[0]]}$")
        ylim = [self.r[axis[1]][0], self.r[axis[1]][-1]]
        ax.set_ylabel(f"${self.axisLabels[axis[1]]}$")
        im = ax.imshow(self[quantity, species][planeSlice].transpose(), origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='equal', vmin=vmin, vmax=vmax, **kwargs)
        return im

    def plot2D(self, quantity, species=0, pos=None, axis=None, suptitle=None, **kwargs):
        if pos is None and self.dim == 3:
            pos = 0
        if axis is None:
            axis = (0, 2) if self.dim == 3 else (0, 1)
        fig, ax = plt.subplots()
        remainingAxis = self._getRemainingAxis(axis)
        remainingAxis = remainingAxis[0] if len(remainingAxis) else None
        posind = self._real2index(pos, remainingAxis)
        im = self._plot2D(ax, quantity, species, posind, axis, **kwargs)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(quantity + "_" + str(species))
        if self.dim == 3:
            ax.set_title(f'${self.axisLabels[remainingAxis]} = {self.rCenter[remainingAxis][posind]}$')
        if suptitle:
            fig.suptitle(suptitle)
        plt.show()

    def _plot2DInteractiveUpdate(self, im, cbar, quantity, species, posind, axis, vmin, vmax):
        planeSlice = self._getSlice(posind, axis)
        im.set_data(self[quantity, species][planeSlice].transpose())
        im.set_clim(vmin, vmax)
        cbar.update_normal(im)

    def plot2DInteractive(self, quantity, species=0, pos=0, axis=(0, 2), vmin=None, vmax=None, **kwargs):
        if self.dim != 3:
            raise NotImplementedError("Interactive plot only works for dim == 3")
        fig = plt.figure()
        axIm = plt.axes((0.0, 0.2, 0.8, 0.7))
        axCbar = plt.axes((0.75, 0.2, 0.03, 0.7))
        remainingAxis = self._getRemainingAxis(axis)[0]
        vmin = np.min(self[quantity, species]) if vmin is None else vmin
        vmax = np.max(self[quantity, species]) if vmax is None else vmax
        im = self._plot2D(axIm, quantity, species, self._real2index(pos, remainingAxis), axis, vmin, vmax, **kwargs)
        cbar = fig.colorbar(im, cax=axCbar)
        cbar.set_label(quantity + "_" + str(species))
        sliderLength = 0.6
        axSliderPlane = plt.axes((0.10, 0.05, sliderLength, 0.03))
        axSliderClimMin = plt.axes((0.9, 0.2, 0.03, 0.7))
        axSliderClimMax = plt.axes((0.95, 0.2, 0.03, 0.7))
        sliderPlane = matplotlib.widgets.Slider(axSliderPlane, f'${self.axisLabels[remainingAxis]}$', 0, self.r[remainingAxis][-1], valinit=pos, closedmax=False)
        # Future matplotlib versions will have RangeSlider
        sliderClimMin = matplotlib.widgets.Slider(axSliderClimMin, '', vmin, vmax, valinit=vmin, orientation='vertical', valfmt='')
        sliderClimMax = matplotlib.widgets.Slider(axSliderClimMax, '', vmin, vmax, valinit=vmax, orientation='vertical', valfmt='')
        sliderPlane.on_changed(lambda var: self._plot2DInteractiveUpdate(im, cbar, quantity, species, self._real2index(var, remainingAxis), axis, sliderClimMin.val, sliderClimMax.val))
        sliderClimMin.on_changed(lambda var: self._plot2DInteractiveUpdate(im, cbar, quantity, species, self._real2index(sliderPlane.val, remainingAxis), axis, var, sliderClimMax.val))
        sliderClimMax.on_changed(lambda var: self._plot2DInteractiveUpdate(im, cbar, quantity, species, self._real2index(sliderPlane.val, remainingAxis), axis, sliderClimMin.val, var))
        plt.show()

    def projectOnto(self, axis):
        self._dataProjected = {s: {k: np.mean(self[k, s], axis=axis) for k in self._data[s]} for s in self._data}
        return self._dataProjected

    def plotProjected(self, onebody, species=(0,), axis=(0, 1), **kwargs):
        self.projectOnto(axis)
        _, ax = plt.subplots()
        remainingAxis = self._getRemainingAxis(axis)[0]
        if type(onebody) is not tuple:
            onebody = (onebody,)
        if type(species) is not tuple:
            species = (species,)
        for f in onebody:
            for s in species:
                ax.plot(self.rCenter[remainingAxis], self._dataProjected[(s)][f], label=f+str(s), **kwargs)
        ax.set_xlabel(f'${self.axisLabels[remainingAxis]}$')
        ax.legend()
        plt.show()


class Radial(Fields):

    def __init__(self, path, stage):
        self.obsType = "radial"
        self.speciesIndices = 2
        super().__init__(path, stage)

    def plot(self, quantities, fig=None, ax=None, **kwargs):
        quantities = np.atleast_1d(quantities)
        showAtEnd = False
        if fig is None and ax is None:
            fig, ax = plt.subplots()
            showAtEnd = True
        for q in quantities:
            ax.plot(self.rCenter[0], self[q], label=q, **kwargs)
        ax.legend()
        if showAtEnd:
            plt.show()


class TwobodyPlanar(Fields):

    def __init__(self, path, stage):
        self.obsType = "twobodyplanar"
        self.speciesIndices = 2
        super().__init__(path, stage)

    def plotRadial1D(self, quantities, z1=0, z2=0, fig=None, ax=None, **kwargs):
        quantities = np.atleast_1d(quantities)
        showAtEnd = False
        if fig is None and ax is None:
            fig, ax = plt.subplots()
            showAtEnd = True
        z1ind = self._real2index(z1, 0)
        z2ind = self._real2index(z2, 1)
        for q in quantities:
            ax.plot(self.rCenter[2], self[q][z1ind, z2ind], label=q, **kwargs)
        ax.legend()
        ax.set_title(f'${self.axisLabels[0]} = {self.rCenter[0][z1ind]}, {self.axisLabels[1]} = {self.rCenter[1][z2ind]}$')
        if showAtEnd:
            plt.show()

    def plotRadial2D(self, quantity: str, z1=0, **kwargs):
        fig, ax = plt.subplots()
        xlim = [self.r[2][0], self.r[2][-1]]
        ax.set_xlabel(f"${self.axisLabels[2]}$")
        ylim = [self.r[1][0], self.r[1][-1]]
        ax.set_ylabel(f"${self.axisLabels[1]}$")
        z1ind = self._real2index(z1, 0)
        im = ax.imshow(self[quantity][z1ind], origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='equal', **kwargs)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(quantity)
        ax.set_title(f'${self.axisLabels[0]} = {self.rCenter[0][z1ind]}$')
        plt.show()


class Clusters(Observables):

    def __init__(self, path, stage):
        self.obsType = "clusters"
        self.speciesIndices = 0
        super().__init__(path, stage)
        try:
            self._data[()]["cumulative"] = np.cumsum(self["clusters"])
        except KeyError:
            pass

    def plot(self, quantity, fig=None, ax=None, log=False, **kwargs):
        showAtEnd = False
        if fig is None and ax is None:
            fig, ax = plt.subplots()
            showAtEnd = True
        if log:
            ax.set_yscale("log")
        ax.plot(self[quantity], **kwargs)
        if showAtEnd:
            plt.show()

