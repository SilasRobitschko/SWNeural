import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import mbd 
import yaml 
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pathlib import Path
import os 
import yaml
import mbd
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.multiprocessing import freeze_support
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

def ensure_no_tf32():
    print("Disabling TF32")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False



def weight_for_loss(dichte, bins=2000):
    # depending on how low the density is, the 
    faktor = np.ones(bins)
    faktor[dichte<0.002]= 0.5
    faktor[dichte<0.0005]= 0.1
    faktor[dichte<0.0001]= 0.0
    return faktor

def losscustom( predictions, targets, loss_factors):
    squared_diff = (predictions - targets) ** 2 #funktioniert alles elementwise
    weighted_diff = squared_diff * loss_factors #Hier jetzt Lossfunction einbauen
    return torch.mean(weighted_diff) #immermean

def wmae( predictions, target, loss_factors):
    return torch.mean(torch.abs(predictions - target) * loss_factors) # genauso wie Loss, nur 

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
    density0 = np.array(run[1].onebody["rho", 0])
    density0 = density0.flatten()
    density1 = np.array(run[1].onebody["rho", 1])
    density1 = density1.flatten()
    vext0 = np.array(run[1].onebody["Eext", 0])
    vext0 = vext0.flatten()
    vext1 = np.array(run[1].onebody["Eext", 1])
    vext1 = vext1.flatten()
    c1_0 = np.log(density0) - mu0/T + vext0/(density0*T)
    c1_1 = np.log(density1) - mu1/T + vext1/(density1*T)
    return density0, density1, c1_0, c1_1, T, Linv




class CPUDataset(Dataset):
    def __init__(self, folderdict, datadict,size, windowsigma=3.5, dz=0.01, bins=2000):
        self.folderdict = folderdict
        self.datadict = datadict 
        self.inputs = []  # Input die 2 rho Profile, 
        self.outputs = []  # Als Output die c1s und eben die lossfactoren
        self.windows = int(windowsigma/dz)
        self.bins = bins
        self.size = size
        # Preload data during initialization

    def generate_data_from_folder(self, folder, bin, case):
        """Simulate data generation from folder and calculate loss factors."""
        # Assuming rundaten and weight functions are imported properly
        rho0ges, rho1ges, c1_0ges, c1_1ges, T, Linv, lfactor_0ges, lfactor_1ges = self.datadict[folder]
        #Window generieren
        rho_0 = rho0ges[bin:bin+2*self.windows+1]
        rho_1 = rho1ges[bin:bin+2*self.windows+1]
        c1_0, c1_1 = c1_0ges[bin], c1_1ges[bin]
        lf0, lf1 = lfactor_0ges[bin], lfactor_1ges[bin]
        #Linv_list.append(Linv)# das ja egal, dann einfach am Ende nicht vergessen
        #T_list.append(T)
        # erstmal Spezies ungetauscht lasssen
        if (case==0):
            inputs = (rho_0, rho_1, T, Linv)  
            outputs = (c1_0, c1_1, lf0, lf1)
        elif (case==1): #Spezies vertauschen          
            inputs = (rho_1, rho_0, T, Linv)  
            outputs = (c1_1, c1_0, lf1, lf0)
        elif (case==2): #x-Achse flippen         
            inputs = (torch.flip(rho_0, dims=[0]), torch.flip(rho_1, dims=[0]), T, Linv)  
            outputs = (c1_0, c1_1, lf0, lf1)
        else: #case3: x-Achse flippen und Spezies vertauschen
            inputs = (torch.flip(rho_1, dims=[0]),torch.flip(rho_0, dims=[0]), T, Linv)
            outputs = (c1_1, c1_0, lf1, lf0)
        return inputs, outputs

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Return CPU tensors - DataLoader will handle transfer to GPU if needed
        folder = self.folderdict[idx][0]
        bin = self.folderdict[idx][1]
        case = self.folderdict[idx][2]
        inputs, outputs = self.generate_data_from_folder(folder, bin, case)
        return inputs, outputs 


def move_batch_to_device(batch, device):
    inputs, outputs = batch
    inputs_gpu = tuple(x.to(device) for x in inputs)
    outputs_gpu = tuple(x.to(device) for x in outputs)
    return inputs_gpu, outputs_gpu

class NetC1(nn.Module):
    def __init__(self, input_size):
        super(NetC1, self).__init__()
        #erstmal 1024 Verbindungen fürs erste layer ausprobieren, macht man damit vielleicht leider, input size = 
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, 2)
        self.softplus = nn.Softplus()
        self.double()

    def forward(self, rho0, rho1, T, Linv):
        x = torch.cat([rho0, rho1, T.unsqueeze(1), Linv.unsqueeze(1)], dim=1)
        x = self.softplus(self.fc1(x))
        x = self.softplus(self.fc2(x))
        x = self.softplus(self.fc3(x))
        x = self.softplus(self.fc4(x))
        x = self.fc_out(x)
        return x


def trainwgpu_cpu(model, train_loader, epochs=100, initial_lr=0.001):
    optimizer = Adam(model.parameters(), lr=initial_lr) #Florians Optimizer kopiert
    
    # Lernrate über Scheduler machen, eig. unnötig aber naja -> Florian kopiert 
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    #auch wieder von Florian die mean absolute error übernehmen 
    # Training history -> kopiert übernehmen
    history = {
        'train_loss': [],
        'train_mae': [],
        'learning_rates': []
    }
    for epoch in range(epochs):
        # das ist das wichtigste der train Befehl 
        model.train()
        #Storen für später
        train_loss = 0
        train_mae = 0
        num_batches = 0
        
        # Training loop # Progressbar von Chatgpt
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            inputs, outputs = move_batch_to_device(batch, 'cuda') #jetzt von cpu auf gpu rüberziehen
            rho_0, rho_1, T, Linv = inputs
            #wir übergeben in den Daten eig. 
            c1_0, c1_1, lfactor_0, lfactor_1 = outputs
            # Vielleich ein bisschen umständlich rausziehen und dann reinziehen mmmh, aber wahrscheinlich nötig, nicht sicher... 
            targets = torch.stack([c1_0, c1_1], dim=1)
            loss_factors = torch.stack([lfactor_0, lfactor_1], dim=1)
            
            # Predicten und Optimizer-Gradienten vorher wieder null setzen
            optimizer.zero_grad()
            predictions = model(rho_0, rho_1, T, Linv)
            
            # weighted loss und mean absolute error berechnen
            loss = losscustom(predictions, targets, loss_factors)
            mae = wmae(predictions, targets, loss_factors) #wie Flo mean berechnen
            
            # Rückwärtsgeben und Updaten der Gewichte , loss.backward() magie ... 
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_mae += mae.item()
            num_batches += 1
            
            # Update progress bar Chatgpt
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'mae': f'{mae.item():.6f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Normieren
        epoch_loss = train_loss / num_batches
        epoch_mae = train_mae / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        # Einfach kopiert übernehmen
        history['train_loss'].append(epoch_loss)
        history['train_mae'].append(epoch_mae)
        history['learning_rates'].append(current_lr)
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {epoch_loss:.4f} - Train MAE: {epoch_mae:.4f} - LR: {current_lr:.6f}')
        
        # Learnrate jetzt erhöhen 
        scheduler.step()
    
    return history

if __name__ == '__main__':
    freeze_support()
    mp.set_start_method('spawn', force=True)
    ensure_no_tf32()    
    root_dir = "/home/silas/Desktop/SWNeural/Notebooks/Runs/Zusammen/"
    # Your existing data loading code
    #jetzt hier vorher Runs mal abchecken: und das alles in einem dict abspeichern, sehr sehr wichtig, root_dir gegeben 
    # theoretisch dieses dict auch in dataset setzbar 
    proot_dir = Path(root_dir)
    folders = [f for f in proot_dir.iterdir() if f.is_dir()] #jetzt ist eben folderpath 

    #wichtigste Element ist eben das folderdict und dementsprechend auch wichtig für die size von dem dict, hier mal gleich für index schreibweise, nötig also bin(2.Stelle) und folder Name (erste Stelle)
    folderdict = {} 
    #datendict mit String des Folderpaths und dann eben Elementen in Reihenfolge 
    # rho0ges, rho1ges, c1_0ges, c1_1ges, T, Linv, lfactor_0ges, lfactor_1ges
    datadict = {}
    windows = 350
    #Alles Folders durchgehen, 
    size = 0
    #erste 
    for folder in folders: 
        if (len(mbd.Simulation(str(folder)).stages[0]) ==2 and int(getattr(mbd.Simulation(str(folder))[1], 'tDiff')) > 500000):
            rho0ges, rho1ges, c1_0ges, c1_1ges, T, Linv = rundata(folder)
            lfactor_0ges = weight_for_loss(rho0ges)
            lfactor_1ges = weight_for_loss(rho1ges)
            rho0ges = torch.from_numpy(rho0ges)
            rho1ges = torch.from_numpy(rho1ges)
            rho0ges =   torch.cat([rho0ges[-windows:], rho0ges, rho0ges[:windows]]) #hier schon Windowen spart vielleicht ein bisschen Zeit 
            rho1ges =   torch.cat([rho1ges[-windows:], rho1ges, rho1ges[:windows]])
            Linv = torch.scalar_tensor(Linv, dtype=torch.float64)
            T = torch.scalar_tensor(T, dtype=torch.float64)
            c1_0ges = torch.from_numpy(c1_0ges)
            c1_1ges = torch.from_numpy(c1_1ges)
            lfactor_0ges = torch.from_numpy(lfactor_0ges)
            lfactor_1ges = torch.from_numpy(lfactor_1ges)
            #Datadict -> Reihenfolge beachten 
            datadict[str(folder)]= rho0ges, rho1ges, c1_0ges, c1_1ges, T, Linv, lfactor_0ges, lfactor_1ges
            #Normales Index Dict  #mit Index und Kriterium dabei! 
            for i in range(2000): #immer 2000 bins
                if (lfactor_0ges[i] ==0.0 and lfactor_1ges[i] == 0.0) or np.isnan(c1_0ges[i]) or np.isnan(c1_1ges[i]):
                    pass
                else:
                    folderdict[size]= [str(folder), i, 0] #als Liste mitgeben, dann eben noch Nummer die anzeigt was passieren soll, eben wenn index
                    folderdict[size+1]= [str(folder), i, 1] 
                    folderdict[size+2]= [str(folder), i, 2]
                    folderdict[size+3]= [str(folder), i, 3]
                    size+=4 #wenn Kriterium erfüllt ist
        else:
            pass
    # Create dataset and dataloader
    dataset = CPUDataset(folderdict, datadict, size)
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    #hier model initialisieren
    inputsize = 4*windows+4 
    model = NetC1(inputsize)
    model.to('cuda')

    #nur als kleiner check 
    batch = next(iter(dataloader))
    inputs, outputs = batch
    for i, tensor in enumerate(inputs):
        print(f"Input {i} dtype:", tensor.dtype)
    for i, tensor in enumerate(outputs):
        print(f"Output {i} dtype:", tensor.dtype)

    #jetzt trainieren 
    trainwgpu_cpu(model, dataloader, epochs=100, initial_lr=0.001)
    torch.save(model, 'full_modelZusammen350.pth')
