import torch
import numpy as np
#---- Custom Loss function ----- 
# A low density causes a noisy c1-value 
# Because of this, reference data with low density gets a decreased factor for backpropagation
def weight_for_loss(density, bins=2000):
    """Calculates weights for the loss function based on density."""
    factor = np.ones(bins)
    factor[density<0.002]= 0.5
    factor[density<0.0005]= 0.1
    factor[density<0.0001]= 0.0
    return factor

def losscustom( predictions, targets, loss_factors):
    """Custom weighted mean squared error loss."""
    squared_diff = (predictions - targets) ** 2 
    weighted_diff = squared_diff * loss_factors 
    return torch.mean(weighted_diff) 

def wmae( predictions, target, loss_factors):
    return torch.mean(torch.abs(predictions - target) * loss_factors) # genauso wie Loss, nur 

#---- Other utilities ----- 
def ensure_no_tf32():
    """Disables TF32 for precision."""
    print("Disabling TF32 for high precision matmul.")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def move_batch_to_device(batch, device):
    """Moves a batch of data to the specified device."""
    inputs, outputs = batch
    inputs_gpu = tuple(x.to(device) for x in inputs)
    outputs_gpu = tuple(x.to(device) for x in outputs)
    return inputs_gpu, outputs_gpu

