from .simulation import *
from .observables import *
from .run import *

import logging
import copy
from scipy.signal import savgol_filter
from scipy.ndimage import zoom


class CustomFlow:

    def __init__(self, path, config, onebodyTarget):
        '''
        Set up the custom flow iteration scheme.
        onebodyTarget must contain the keys "rho" and "v", which are the target density and velocity profiles.
        '''
        self.path = path
        self.systemConfig = config["System"]
        self.integratorConfig = config["Integrator"]
        self.initParticlesConfig = config["InitParticles"]
        self.T = self.systemConfig["T"]
        self.gamma = config["InitParticles"][0]["lattice"]["particleProperties"]["gamma"]
        self.onebodyTarget = onebodyTarget
        self.onebodyActual = copy.deepcopy(onebodyTarget)
        self.fextDoFilter = True
        self.fextFilterWindow = 7 # must be odd
        self.fextDoUpsample = True
        self.fextUpsampleFactor = 10
        # First estimate of Fext with ideal gas
        for d in ["x", "y", "z"]:
            self.onebodyActual.data[0]["Fint"+d] = np.zeros(self.onebodyActual.shape)

    def runIteration(self, iteration, stepsEquilibrate, stepsSimulate, ensemble=None, cluster=False):
        iterationDir = f"{self.path}/customflow{iteration}"
        self._writeConfig(iteration, stepsEquilibrate, stepsSimulate, ensemble)
        runMBD(f"-c{iterationDir}.yaml -p{iterationDir}", ensemble=ensemble, cluster=cluster)
        iterationEnsemble = Ensemble(iterationDir, stageIds=[1])
        self.onebodyActual = iterationEnsemble.average.onebody
        return iterationEnsemble

    def _writeConfig(self, iteration, stepsEquilibrate, stepsSimulate, ensemble):
        self.fextBins, self.fextNewx, self.fextNewy, self.fextNewz = self._calcFext()
        config = {}
        config["System"] = self.systemConfig
        config["Integrator"] = self.integratorConfig
        if iteration == 0:
            config["InitParticles"] = self.initParticlesConfig
        else:
            config["InitParticles"] = [{
                "file": {
                    "filename": f"{self.path}/customflow{iteration-1}{'/run0' if ensemble else ''}/particlesLast.dat"
                }
            }]
        config["Stages"] = [
            {"steps": stepsEquilibrate},
            {"steps": stepsSimulate,
             "Observables": {
                 "Onebody": {
                     "bins": list(self.onebodyTarget.shape),
                     "keys": ["1", "Fintx", "Finty", "Fintz", "Fextx", "Fexty", "Fextz", "rRandx", "rRandy", "rRandz"]
                 }
             }
             }
        ]
        config["System"]["interaction"]["external"] = {
            "ExternalTable": {
                "bins": self.fextBins,
                "fextx": self.fextNewx.flatten().tolist(),
                "fexty": self.fextNewy.flatten().tolist(),
                "fextz": self.fextNewz.flatten().tolist(),
            }
        }
        with open(f"{self.path}/customflow{iteration}.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=True)

    def _calcFext(self):
        for d in ["x", "y", "z"]:
            if "vRand"+d in self.onebodyTarget.data[0] and "Fint"+d in self.onebodyActual.data[0]:
                self.onebodyActual.data[0]["fextNew"+d] = - self.gamma * self.onebodyTarget["vRand"+d] - self.onebodyActual["Fint"+d] / self.onebodyTarget["rho"] + self.gamma * self.onebodyTarget["v"+d]
            else:
                self.onebodyActual.data[0]["fextNew"+d] = np.zeros(self.onebodyActual.shape)
        result = {}
        upsamplingFactors = [self.fextUpsampleFactor if s > 1 else 1 for s in self.onebodyActual.shape]
        for d in ["x", "y", "z"]:
            maybeFiltered = savgol_filter(self.onebodyActual["fextNew"+d], self.fextFilterWindow // 2 * 2 + 1, 3) if self.fextDoFilter else self.onebodyActual["fextNew"+d]
            result["fextNew"+d] = zoom(maybeFiltered, upsamplingFactors, prefilter=False, order=1, mode="wrap") if self.fextDoUpsample > 1 else maybeFiltered
        bins = list(result["fextNewx"].shape)
        return bins, result["fextNewx"], result["fextNewy"], result["fextNewz"]


class AdiabaticReference:

    def __init__(self, noneqEnsemble, runPath, method='customflow'):
        '''
        Construct an adiabatic reference system from the results of a nonequilibrium simulation at noneqPath
        '''
        from pathlib import Path
        self.noneqEnsemble = noneqEnsemble
        runPath = Path(runPath)
        if runPath.exists():
            logging.warning("Path for adiabatic reference construction already exists. If you continue, existing iterations will be reused.")
            val = input("Continue? [Y/n] ")
            if not (val == "y" or val == "Y" or val == ""):
                raise FileExistsError("Path for adiabatic reference constrution already exists and shall not be reused.")
        runPath.mkdir(parents=True, exist_ok=True)
        self.runPath = runPath.resolve()
        noneqConfig = noneqEnsemble.config
        self.onebodyTarget = noneqEnsemble.average.onebody
        if method == "customflow":
            # We want an equilibrium system with v = 0
            for d in ["x", "y", "z"]:
                self.onebodyTarget.data[0]["v"+d] = np.zeros(self.onebodyTarget.shape)
            self.referenceRunner = CustomFlow(self.runPath, noneqConfig, self.onebodyTarget)

    def run(self, iterations):
        for i in range(iterations):
            self.runIteration(i)

    def runIteration(self, iteration, stepsEquilibrate=1000, stepsSimulate=10000, ensemble=None, cluster=False) -> Ensemble:
        return self.referenceRunner.runIteration(iteration, stepsEquilibrate, stepsSimulate, ensemble, cluster)
