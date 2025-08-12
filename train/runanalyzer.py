import mbd
import yaml
import numpy as np

def read_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)
    
def rundata(path):
    run = mbd.Simulation(str(path))
    config = read_yaml(str(path) + "/config.yaml")
    T = config['System']['T']
    mu0 = config['System']['mu']
    Linv =  1.0/ config['System']['L'][0]
    rho0 = np.array(run[1].onebody["rho"])
    rho0 = rho0.flatten()
    vext0 = np.array(run[1].onebody["Eext"])
    vext0 = vext0.flatten()
    c1_0 = np.log(rho0) - mu0/T + vext0/(rho0*T)
    return rho0, c1_0, T, Linv

