import torch
from pathlib import Path
import matplotlib.pyplot as plt
#import from the utility script
#important function calc_rho which uses fix point iterations for the density profile
#take a look at it in utils.py if you want to implement your own calculations of the density profile
from utils import ensure_no_tf32, calc_rho
from model import NetC1

if __name__ == '__main__':
    # --- PARAMETERS: CHANGE IF YOU WANT TO ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelname = "original_model.pth"
    windows = 350 #make sure the window-size is consisten with the model

    #--- Setup --- 
    ensure_no_tf32()    
    project_dir = Path(__file__).resolve().parent.parent
    run_dir = project_dir / "simdata" / "cluster_03_12_24" 
    model_dir = project_dir/ "models"
    results_dir = project_dir / "results"

    c1model = torch.load(model_dir / modelname, weights_only=False)
    c1model.eval()
    c1model.to(device)

    #Don't change this dx unless you use different training data!
    dx = 0.01 # the discretization step of the profiles (profiles vary in z-direction constant in x/y direction) 
    
    #--- Setup for the Alpha Beta Interfacial Profile --- 

    #initialize the setup
    T=0.93
    L= 100 
    bins = int(L/dx)
    rho0init = torch.zeros(bins, dtype=torch.float64)
    rho1init = torch.zeros(bins, dtype=torch.float64)
    rhomean = 0.375
    # asymmetric initial density 
    rho0init[:5000] = 0.55
    rho1init[:5000] = 0.10
    rho0init[5000:] = 0.05
    rho1init[5000:] = 0.05
    # calculate equilibrium density profile with the constraint of keeping the total density fixed
    # between liquid vapor --> two phases arise in order to fulfill constraint
    xs, rho0alphabeta, rho1alphabeta = calc_rho(model=c1model, T=T, L=L, rho0_init=rho0init, rho1_init=rho1init, rho_mean=rhomean, device=device)

    # Now plot and save the results 
    plt.figure(figsize=(4, 3.5), dpi=300)

    # Plot the data with a shift
    shift = 2500
    plt.plot(xs, torch.roll(rho0alphabeta, shifts=shift), label=r"$\rho_0 \sigma^3$", linewidth=2)
    plt.plot(xs, torch.roll(rho1alphabeta, shifts=shift), label=r"$\rho_1 \sigma^3$", linewidth=2, linestyle="dashed")
    plt.ylim(0)
    plt.xlim(0,50)
    plt.xlabel(r"$z / \sigma$", fontsize=13)
    plt.legend(fontsize=12, loc="best")
    plt.title(r"Demixed-Vapor Interface ($k_B T/\epsilon=0.93$)", fontsize=14)
    plt.savefig( results_dir / "Alpha_Beta_at_T_093.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    

    ''' 
    # --- Setup for the Beta Gamma Interfacial Profile --- 
    # Remove comment to run this too! 
    T=0.93
    L= 100 
    bins = int(L/dx)
    rho0init = torch.zeros(bins, dtype=torch.float64)
    rho1init = torch.zeros(bins, dtype=torch.float64)
    rhobulk = 0.67
    # symmetric but species flipped density 
    rho0init[:5000] = 0.58
    rho1init[:5000] = 0.09
    rho0init[5000:] = 0.09
    rho1init[5000:] = 0.58
    # calculate equilibrium density profile with the constraint of keeping the bulk density (around 25 sigma and 75 sigma) fixed
    # liquid-liquid phase transition arises (with majority 0 phase and majority 1 phase)
    xs, rho0betagamma, rho1betagamma = calc_rho(model=c1model, T=T, L=L, rho0_init=rho0init, rho1_init=rho1init, rho_bulk=rhobulk, device=device)    
    
    
    # Now plot and save the results 
    plt.figure(figsize=(4, 4), dpi=300)

    # Plot the data with a shift
    shift = 2500
    plt.plot(xs, torch.roll(rho0betagamma, shifts=shift), label=r"$\rho_0 \sigma^3$", linewidth=2)
    plt.plot(xs, torch.roll(rho1betagamma, shifts=shift), label=r"$\rho_1 \sigma^3$", linewidth=2)
    plt.ylim(0)
    plt.xlim(0,50)
    plt.xlabel(r"$z / \sigma$", fontsize=13)
    plt.legend(fontsize=12, loc="best")
    plt.title(r"Liquid-Liquid Interface ($k_B T/\epsilon=0.93$)", fontsize=14)
    plt.savefig( results_dir / "Beta_Gamma_at_T_093_and_rho_067.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    '''
