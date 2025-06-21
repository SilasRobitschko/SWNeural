import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.multiprocessing import freeze_support, set_start_method
from pathlib import Path
from tqdm import tqdm
import numpy as np
#custom library 
import mbd

#import from other files
from runanalyzer import rundata 
from dataset import CPUDataset
from model import NetC1
from utils import ensure_no_tf32, losscustom, wmae, move_batch_to_device,weight_for_loss 

#Net-Input two density windows(around 1 bins), Temperature, inverse Boxsize
#Net-Output two scaler c1 values 
def trainwgpu_cpu(model, train_loader, epochs=100, initial_lr=0.001, device='cuda'):
    '''Training Loop for the Neural Net: 
    - model: The model one wants to train
    - train_loader: The dataloader'''
    #Adam optimizer
    optimizer = Adam(model.parameters(), lr=initial_lr) 
    #Scheduler for decreasing Learning rate 
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    history = {
        'train_loss': [],
        'train_mae': [],
        'learning_rates': []
    }
    for epoch in range(epochs):
        model.train()
        train_loss, train_mae, num_batches = 0, 0, 0
        #Cosmetic Progress Bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            inputs, outputs = move_batch_to_device(batch, device) #move to device

            rho_0, rho_1, T, Linv = inputs
            c1_0, c1_1, lfactor_0, lfactor_1 = outputs

            #only c1_0, c1_1 are actual model outputs 
            targets = torch.stack([c1_0, c1_1], dim=1)
            loss_factors = torch.stack([lfactor_0, lfactor_1], dim=1)

            optimizer.zero_grad()
            #now use model for inference
            predictions = model(rho_0, rho_1, T, Linv)
            
            #Custom loss function
            loss = losscustom(predictions, targets, loss_factors)
            mae = wmae(predictions, targets, loss_factors)
            
            #Backpropagation and Learning
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += mae.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'mae': f'{mae.item():.6f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        epoch_loss = train_loss / num_batches
        epoch_mae = train_mae / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(epoch_loss)
        history['train_mae'].append(epoch_mae)
        history['learning_rates'].append(current_lr)
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {epoch_loss:.4f} - Train MAE: {epoch_mae:.4f} - LR: {current_lr:.6f}')
        
        #Increase Scheduler
        scheduler.step()
    
    return history

#--- Main Function --- 
if __name__ == '__main__':

    #--- Setup --- 
    freeze_support()
    set_start_method('spawn', force=True)
    ensure_no_tf32()    
    
    project_dir = Path(__file__).resolve().parent.parent
    run_dir = project_dir / "simdata" / "cluster_03_12_24" 
    model_dir = project_dir / "models"

    # --- PARAMETERS: CHANGE IF YOU WANT TO ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelname = "new_model.pth"
    # Windowsize for the density windows is 350 bins corresponding to 3.5 sigma
    windows = 350
    #Training Parameters
    epochs = 100
    learningrate = 0.001
    batchsize = 256 
    num_of_workers = 4

    # --- Load the reference data ---

    #Collect all the folders for the rundata 
    prun_dir = Path(run_dir)
    folders = [f for f in prun_dir.iterdir() if f.is_dir()] 
    #folderdict: {indexnumber:(foldername, bin(x-position), augmentation-case)}
    folderdict = {} 
    index = 0 #index runs from 0 to number of bins*number of runs*number of augmentation cases
    #datadict: {foldername: (rho0full, rho1full, c1_0full, T, Linv, lfactor0full, lfactor1full)}   with full indicating an array of 2000 bins
    datadict = {}

    bins = 2000 #all profiles are discretized with 2000 bins
    for folder in folders: 
        if (len(mbd.Simulation(str(folder)).stages[0]) ==2 and int(getattr(mbd.Simulation(str(folder))[1], 'tDiff')) > 500000):
            rho0ges, rho1ges, c1_0ges, c1_1ges, T, Linv = rundata(folder)
            lfactor_0ges = weight_for_loss(rho0ges)
            lfactor_1ges = weight_for_loss(rho1ges)
            rho0ges = torch.from_numpy(rho0ges)
            rho1ges = torch.from_numpy(rho1ges)
            # Use periodic boundary conditions for density windows
            rho0ges =   torch.cat([rho0ges[-windows:], rho0ges, rho0ges[:windows]]) 
            rho1ges =   torch.cat([rho1ges[-windows:], rho1ges, rho1ges[:windows]])
            Linv = torch.scalar_tensor(Linv, dtype=torch.float64)
            T = torch.scalar_tensor(T, dtype=torch.float64)
            c1_0ges = torch.from_numpy(c1_0ges)
            c1_1ges = torch.from_numpy(c1_1ges)
            lfactor_0ges = torch.from_numpy(lfactor_0ges)
            lfactor_1ges = torch.from_numpy(lfactor_1ges)
            #save this into the datafolder
            datadict[str(folder)]= rho0ges, rho1ges, c1_0ges, c1_1ges, T, Linv, lfactor_0ges, lfactor_1ges

            for i in range(bins): 
                #already discard "bad" datapoints
                if (lfactor_0ges[i] ==0.0 and lfactor_1ges[i] == 0.0) or np.isnan(c1_0ges[i]) or np.isnan(c1_1ges[i]):
                    pass
                else:
                    folderdict[index]= [str(folder), i, 0] 
                    folderdict[index+1]= [str(folder), i, 1] 
                    folderdict[index+2]= [str(folder), i, 2]
                    folderdict[index+3]= [str(folder), i, 3]
                    index+=4 
        else:
            pass

    # --- Create dataset and dataloader ---
    totalsize = index
    dataset = CPUDataset(folderdict, datadict, totalsize)
    dataloader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        pin_memory=True,
        num_workers=num_of_workers
    )

    #--- Initialize model and train ---
    inputsize = 4*windows+4 
    model = NetC1(inputsize)
    model.to(device)

    trainwgpu_cpu(model, dataloader, epochs=epochs, initial_lr=learningrate, device=device)
    torch.save(model, model_dir / modelname)
