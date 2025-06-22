import torch
import math
def ensure_no_tf32():
    """Disables TF32 for precision."""
    print("Disabling TF32 for high precision matmul.")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def c1_profiles(model, rho0, rho1, T, Lxy,windows=350, batch_size=4096, device=torch.device("cuda")):
    """
    Predicts the c1 Profiles (c1_0 and c1_1) for large density profiles by batched inference on a model .

    It works the following way:
    1. Create sliding windows of the density profiles, one window corresponds to one bin for the c1_0 
    2. Iterating through these windows in batches and moving a batch to the GPU (device) for prediction
    3. Collecting the result and moving them back to the CPU 

    Args:
        model: The trained pytorch model
        rho0: The 1D density profile for species 0 as torch.Tensor.
        rho1: The 1D density profile for species 1 as torch.Tensor.
        T: Temperature
        Lxy: Lateral box size. If None: automatically is set to infinity
        windows: The half-size of the sliding window (total window size is 2*windows+1).
        batch_size (int): The number of windows to process in a single batch on the GPU.
        device: The device to run inference on.

    """
    model.eval() 
    # Disable gradient calculations for inference crucially saving memory 
    with torch.no_grad():
        rho0_cpu = rho0.detach().cpu()
        rho1_cpu = rho1.detach().cpu()
        # Employ periodic boundary conditions for the windows
        rho0_padded = torch.cat([rho0_cpu[-windows:], rho0_cpu, rho0_cpu[:windows]])
        rho1_padded = torch.cat([rho1_cpu[-windows:], rho1_cpu, rho1_cpu[:windows]])
        # Create all the density windows screenshots 
        rho0_windows = rho0_padded.unfold(0, 2 * windows + 1, 1)
        rho1_windows = rho1_padded.unfold(0, 2 * windows + 1, 1)
        
        total_size = rho0_windows.size(0)
        # Prepare c1 profiles
        c1_0 = torch.zeros(total_size, device='cpu', dtype=torch.float64)
        c1_1 = torch.zeros(total_size, device='cpu', dtype=torch.float64)

        if Lxy == None: 
            Linv = torch.tensor(0, dtype=torch.float64,device=device)
        else:
            Linv = torch.tensor(1.0/Lxy, dtype=torch.float64, device=device)

        T = torch.tensor(T, dtype=torch.float64, device=device)
        
        # ---  Batched Inference Loop ---
        for i in range(0, total_size, batch_size):
            end_idx = min(i + batch_size, total_size)
            current_batch_size = end_idx - i

            # Move the current batch of windows to the specified device, nonblocking=True for good handling
            batch_rho0 = rho0_windows[i:end_idx].to(device, non_blocking=True)
            batch_rho1 = rho1_windows[i:end_idx].to(device, non_blocking=True)
            
            # Expand scalar parameters to match the batch size
            T_batch = T.expand(current_batch_size)
            Linv_batch = Linv.expand(current_batch_size)
            
            # Run model prediction on the batches
            predictions = model(batch_rho0, batch_rho1, T_batch, Linv_batch)
            
            # Store results back on the CPU
            c1_0[i:end_idx] = predictions[:, 0].cpu()
            c1_1[i:end_idx] = predictions[:, 1].cpu()

    return c1_0, c1_1


# Function to calculate the density profiles from the c1 using Fixpoint Iterations
def calc_rho(model, T, L,Lxy=None, rho0_init=None, rho1_init=None,Vext0=None, Vext1=None,
    mu=None, mu_delta=0,rho_mean=None,rho_bulk=None, dx=0.01, alpha=0.075, tol=1e-5, max_iter=100000, print_every=2000,device=torch.device('cuda')):
    """
    Determines the equilibrium density profile by solving the Euler-Lagrange equation self-consistently with a mixed Picard iteration.
    Change one of the three constraints to achieve the desired results: 
    - mu and mu_delta: mu = chemical potential for species 0 mu + mu_delta = chemical potential for species 1
    - rho_mean: the total density (across both species) in the system (used for liquid-vapor interfaces)
    - rho_bulk: the target total density in a small, defined bulk region (used for liquid-liquid interfaces)

    Args:
        model: The trained neural network for the correlation functional.
        T: The system temperature.
        L: The length of the system box across which the density profile can vary
        Vext0, Vext1: External potential profiles for species 0 and 1 as torch.tensor -> If None, chosen as 0 
        mu : The chemical potential for the grand canonical ensemble.
        rho_mean : The target mean total density for the canonical ensemble.
        rho_bulk : The target density in a defined bulk region.
        Lxy : The system's box size perpendicular to the profile.
        mu_delta: The chemical potential difference (mu1 = mu0 + mu_delta).
        dx : The spatial discretization interval. Must match the model.
        rho0_init, rho1_init (torch.Tensor, optional): Initial density profiles -> If None, two constant profiles
        alpha: The initial Picard mixing parameter (learning rate).
        tol: The tolerance for convergence.
        max_iter: The maximum number of iterations.
        print_every: How often to print progress. Set to 0 for no prints.
        device: The PyTorch device ('cuda' or 'cpu') to use for inference
    """

    # Check if only one of the constraints is active
    constraints_val = sum(arg is not None for arg in [mu, rho_mean, rho_bulk])
    if constraints_val != 1:
        raise ValueError("Specify exactly one of 'mu', 'rho_mean', or 'rho_bulk'.")

    xs = torch.arange(dx / 2, L, dx, device=device, dtype=torch.float64)
    num_bins = len(xs)

    if rho0_init is None:
        rho0_init = torch.full((num_bins,), 0.20, dtype=torch.float64)
    if rho1_init is None:
        rho1_init = torch.full((num_bins,), 0.25, dtype=torch.float64)
    if Vext0 is None:
        Vext0 = torch.full((num_bins,), 0, dtype=torch.float64)
    if Vext1 is None:
        Vext1 = torch.full((num_bins,), 0, dtype=torch.float64)
    

    # Pre-calculate indices for the 'rho_bulk' constraint mode
    if rho_bulk is not None:
        i_center1 = int(L / 4 / dx)
        i_center2 = int(3 * L / 4 / dx)
        i_range   = int(L / 100 / dx) # A 1% region around the quarter-points where we expect the profile to be in bulk 
        bulk_indices = torch.cat([
            torch.arange(i_center1 - i_range, i_center1 + i_range),
            torch.arange(i_center2 - i_range, i_center2 + i_range)
        ])

    # --- Self consisten Fixpoint Iteration Loop---
    for i in range(1, max_iter + 1):

        # --- Symmetrized c1 Calculation ---
        # This enforces physical symmetries, namely particle exchange, spatial inversion symmetry
        c1_0_std, c1_1_std = c1_profiles(model, rho0, rho1, T, Lxy, device=device)
        c1_1_swap, c1_0_swap = c1_profiles(model, rho1, rho0, T, Lxy, device=device)
        c1_0_flip, c1_1_flip = c1_profiles(model, rho0.flip(0), rho1.flip(0), T, Lxy, device=device)
        c1_1_both, c_0_both = c1_profiles(model, rho1.flip(0), rho0.flip(0), T, Lxy, device=device)
        c1_0 = 0.25 * (c1_0_std + c1_0_swap + c1_0_flip.flip(0) + c_0_both.flip(0))
        c1_1 = 0.25 * (c1_1_std + c1_1_swap + c1_1_flip.flip(0) + c1_1_both.flip(0))
        
        # Calculate the unnormalized new densities from the Euler-Lagrange equation
        rho0_new_unnorm = torch.exp(-Vext0 / T + c1_0)
        rho1_new_unnorm = torch.exp(-Vext1 / T + c1_1)

        # --- Implementation of the Constraints ---
        if mu is not None:
            mu0 = mu
            mu1 = mu0 + mu_delta
            rho0_new = rho0_new_unnorm * math.exp(mu0 / T)
            rho1_new = rho1_new_unnorm * math.exp(mu1 / T)
        
        elif rho_mean is not None:
            # rescale to math mean density 
            current_mean_density = (torch.sum(rho0_new_unnorm) + torch.sum(rho1_new_unnorm)) * dx / L / 2
            scaling_factor = rho_mean / current_mean_density 
            rho0_new = rho0_new_unnorm * scaling_factor
            rho1_new = rho1_new_unnorm * scaling_factor
        
        elif rho_bulk is not None:
            # rescale to match bulk density
            current_bulk_density = (torch.mean(rho0_new_unnorm[bulk_indices]) + torch.mean(rho1_new_unnorm[bulk_indices])) / 2
            scaling_factor = rho_bulk / current_bulk_density 
            rho0_new = rho0_new_unnorm * scaling_factor
            rho1_new = rho1_new_unnorm * scaling_factor
        
        # --- Picard Mixing---
        rho0 = (1 - alpha) * rho0 + alpha * rho0_new
        rho1 = (1 - alpha) * rho1 + alpha * rho1_new

        delta0 = torch.max(torch.abs(rho0 - rho0_new))
        delta1 = torch.max(torch.abs(rho1 - rho1_new))

        if print_every > 0 and i % print_every == 0:
            print(f"Iteration {i}: delta0 = {delta0}, delta1 = {delta1}")
        
        if not (torch.isfinite(delta0) and torch.isfinite(delta1)):
            print(f"Iteration {i}: Diverged with non-finite delta (delta0={delta0}, delta1={delta1})")
            return xs, None, None

        # Convergence Criteria
        if delta0 < tol and delta1 < tol:
            print(f"Converged after {i} iterations (delta0={delta0}, delta1={delta1})")
            return xs, rho0, rho1
        
    print(f"Did not converge after {max_iter} iterations (delta0={delta0}, delta1={delta1})")
    return xs, None, None