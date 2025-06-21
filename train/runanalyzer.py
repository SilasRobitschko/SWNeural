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
    mu0 = config['System']['mu0']
    mu1 = mu0
    Linv =  1.0/ config['System']['L'][0]
    rho0 = np.array(run[1].onebody["rho", 0])
    rho0 = rho0.flatten()
    rho1 = np.array(run[1].onebody["rho", 1])
    rho1 = rho1.flatten()
    vext0 = np.array(run[1].onebody["Eext", 0])
    vext0 = vext0.flatten()
    vext1 = np.array(run[1].onebody["Eext", 1])
    vext1 = vext1.flatten()
    c1_0 = np.log(rho0) - mu0/T + vext0/(rho0*T)
    c1_1 = np.log(rho1) - mu1/T + vext1/(rho1*T)
    return rho0, rho1, c1_0, c1_1, T, Linv