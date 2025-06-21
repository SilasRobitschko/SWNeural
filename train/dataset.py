import torch
from torch.utils.data import Dataset
# Load the data on the spot with the CPU, keep preprocessing in the dataset at the minimum to increase speed
class CPUDataset(Dataset):
    def __init__(self, folderdict, datadict,size, windowsigma=3.5, dz=0.01, bins=2000):
        self.folderdict = folderdict #folderdict: {indexnumber:(foldername, bin(x-position), augmentation-case)}
        self.datadict = datadict  #datadict: {foldername: (rho0full, rho1full, c1_0full, T, Linv, lfactor0full, lfactor1full)}   with full indicating an array of 2000 bins
        self.inputs = []  
        self.outputs = []  
        self.windows = int(windowsigma/dz)
        self.bins = bins
        self.size = size

    def generate_data_from_folder(self, folder, bin, case):
        """Retrieve data from presaved datadict and augment data"""
        rho0ges, rho1ges, c1_0ges, c1_1ges, T, Linv, lfactor_0ges, lfactor_1ges = self.datadict[folder]
        # Generate a density window around the specific bin
        rho_0 = rho0ges[bin:bin+2*self.windows+1]
        rho_1 = rho1ges[bin:bin+2*self.windows+1]
        # This density window corresponds to the c1 value at this bin 
        c1_0, c1_1 = c1_0ges[bin], c1_1ges[bin]
        lf0, lf1 = lfactor_0ges[bin], lfactor_1ges[bin]

        # Data Augmentation
        if (case==0):
            inputs = (rho_0, rho_1, T, Linv)  
            outputs = (c1_0, c1_1, lf0, lf1)
        elif (case==1): #Species flip       
            inputs = (rho_1, rho_0, T, Linv)  
            outputs = (c1_1, c1_0, lf1, lf0)
        elif (case==2): #x-axis flip         
            inputs = (torch.flip(rho_0, dims=[0]), torch.flip(rho_1, dims=[0]), T, Linv)  
            outputs = (c1_0, c1_1, lf0, lf1)
        else: #x-axis Flip and species flip
            inputs = (torch.flip(rho_1, dims=[0]),torch.flip(rho_0, dims=[0]), T, Linv)
            outputs = (c1_1, c1_0, lf1, lf0)
        
        return inputs, outputs 

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Random index -> corresponds to 
        folder = self.folderdict[idx][0]
        bin = self.folderdict[idx][1]
        case = self.folderdict[idx][2]
        inputs, outputs = self.generate_data_from_folder(folder, bin, case)
        #inputs: rho0window, rho1window, T, Linv   outputs: c1_0, c1_1, Lossfactor0, Lossfactor1
        return inputs, outputs 