import torch
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import NMF 

def robust_nmf_breg(X: np.ndarray, n_comp: int, iterations: int, λ: float, tol: float, device='cpu'):
    # PyTorch adaptation of the algorithm 
    # in the paper https://pmc.ncbi.nlm.nih.gov/articles/PMC8541511/
    
    # Get the shape of the data
    X_height, X_width, X_frames = X.shape
    
    # Reshape the video matrix
    X = X.reshape(X_height * X_width, X_frames)
    
    # Precalculate W and H using NMF from sklearn
    nmf = NMF(n_components=n_comp, random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_
    
    # Initalisation
    X = torch.tensor(X, dtype=torch.float32, device=device)
    S = torch.zeros_like(X, dtype=torch.float32, device=device)
    p = torch.zeros_like(X, dtype=torch.float32, device=device)
    W = torch.tensor(W, dtype=torch.float32, device=device)
    H = torch.tensor(H, dtype=torch.float32, device=device)
    
    # define the objective func 
    loss = lambda W, H, S: .5*torch.norm(X - W @ H - S, 'fro') ** 2 # reconstruction loss
    reg = lambda S: torch.norm(S, p=1) # l1 matrix reg
    objective = lambda W, H, S: loss(W, H, S) + λ * reg(S)
    
    # define the sigma clamp
    σ = lambda S, λ: torch.where(S > λ, S - λ, 0)
    
    # Initialize objective list 
    o1 = -100 # some value does not matter
    o2 = None
        
    # Main Loop
    for i in range(iterations):
        # Step 1 in Algorithm 1: Update S
        S = X - W @ H + λ * p
        S = σ(S, λ) # enforce positivity constraint
        
        # Step 2 in Algorithm 1: Update p
        p = p + (1 / λ) * ((X - W @ H) - S)
        
        # Step 3, 4 in Algorithm 1: Update W and H
        SX = S - X 
        W[W < 1e-10] = 1e-10 # for numerical stability
        H[H < 1e-10] = 1e-10 # for numerical stability
        SXH = SX @ H.t()
        WSX = W.t() @ SX
        WH_grad_W = torch.abs(SXH) - SXH
        WH_grad_H = torch.abs(WSX) - WSX
        W *= (WH_grad_W / (2 * (W @ (H @ H.t()))))
        H *= (WH_grad_H / (2 * (W.t() @ (W @ H))))
        
        # Normalise W and ajust H 
        norm_W = torch.sqrt(torch.sum(W ** 2, dim=0))
        norm_mask = norm_W > 0
        W[:, norm_mask] /= norm_W[norm_mask]
        H[norm_mask, :] *= norm_W[norm_mask].reshape(-1, 1)
        
        # Objective computation
        current_obj = objective(W, H, S).item()
        o2 = current_obj
        
        # if less than tolerance
        if abs((o2 - o1) / o1) < tol:
            break # early stopping
        
        # relabel
        o1 = o2 
        o2 = None  
        
    return W.cpu().detach().numpy(), H.cpu().detach().numpy(), S.reshape(X_height, X_width, X_frames).cpu().detach().numpy()
    