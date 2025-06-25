import torch
from torch import nn

class NetC1(nn.Module):
    ''' Basic Linear-Neural model with softplus activation function : inputsizex1024x512x512x512x2:'''
    #The inputsize is determined by the windowsize of rho0 (rho1)
    def __init__(self, input_size):
        super(NetC1, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, 2)
        self.softplus = nn.Softplus()
        self.double()  # Set model parameters to float64

    def forward(self, rho0, rho1, T, Linv):
        # Ensure T and Linv are correctly shaped for batch processing
        x = torch.cat([rho0, rho1, T.unsqueeze(1), Linv.unsqueeze(1)], dim=1)
        x = self.softplus(self.fc1(x))
        x = self.softplus(self.fc2(x))
        x = self.softplus(self.fc3(x))
        x = self.softplus(self.fc4(x))
        x = self.fc_out(x)
        return x