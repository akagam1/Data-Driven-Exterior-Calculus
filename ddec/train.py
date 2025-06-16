import matplotlib.pyplot as plt
import numpy as np
from DDEC_Net import DDECModel, PerturbNet
import torch
from tqdm import tqdm
from utils import *

def plot_contour(arr1, arr2, title1='Array 1', title2='Array 2', cmap='viridis'):
    """
    Plot two 2D arrays as contour plots side by side.
    
    Args:
        arr1 (np.ndarray or torch.Tensor): First 2D array.
        arr2 (np.ndarray or torch.Tensor): Second 2D array.
        title1 (str): Title for the first plot.
        title2 (str): Title for the second plot.
        cmap (str): Colormap to use (default 'viridis').
    """
    if hasattr(arr1, 'detach'):  # convert torch.Tensor to numpy if needed
        arr1 = arr1.detach().cpu().numpy()
    if hasattr(arr2, 'detach'):
        arr2 = arr2.detach().cpu().numpy()
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, arr, title in zip(axes, [arr1, arr2], [title1, title2]):
        n, m = arr.shape
        x = np.linspace(0, 1, m)
        y = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, y)
        contour = ax.contourf(X, Y, arr, 20, cmap=cmap)
        fig.colorbar(contour, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def train_main(N, alpha, iter, tol, epsilon, in_dim, out_dim, epochs, problem_type, lr,show_plot=False):

    batches, d0, d1 = dataset_generation(N, problem_type=problem_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d0 = d0.to(device=device)
    d1 = d1.to(device=device)
    properties = {'d0': d0, 'd1': d1}

    model = DDECModel(iter, tol, in_dim, out_dim, properties, cochain=0)

    perturb = PerturbNet((N)*(N), 50, device=device)
    optimizer = torch.optim.Adam(list(model.parameters())+list(perturb.parameters()), lr=lr)
    criterion = torch.nn.MSELoss()
    losses = []
    epochs = epochs

    #u = make_u(N)
    u = abs(torch.randn((N)*(N), dtype=torch.float64, requires_grad=True))

    print(f"Using device: {device}")
    model = model.to(device)
    u = u.to(device)
    perturb = perturb.to(device)

    # with torch.no_grad():
    #     for p in [model.B0_vals, model.B1_vals, model.B2_vals, model.D0_vals, model.D1_vals, model.D2_vals]:
    #         p.clamp_(0, 1)

    with tqdm(total= epochs, desc="Training", unit="epoch", colour='green') as pbar:
        for epoch in range(epochs):
            perturb_contribution = 0
            for X in batches:
                f, phi_faces,alpha = X
                f = f.to(device)
                phi_faces = phi_faces.to(device)
                model.phi_faces = phi_faces.clone().detach().requires_grad_(True)
                model.f = f.clone().detach().requires_grad_(True)
                optimizer.zero_grad()
                with torch.enable_grad():
                    u_det = u.detach().requires_grad_(True)
                    u_it = model(u_det, f)
                    f_re = f.view(-1,1)
                    error = perturb(f_re).squeeze(-1).squeeze(-1)
                    u_est = u_it + epsilon*error
                    perturb_contribution += error.abs().mean().item()

                    l = model.adj_loss

                    loss = criterion(u_est, phi_faces) + l
                    loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()

                losses.append(loss.item())

            pbar.set_postfix({
        "loss": f"{loss.item():.8f}"
    })
            pbar.update(1)
    print("")
    print("="*50)
    print(f"Training completed after {epochs} epochs with final loss: {loss.item()}")

    if problem_type == 'D2':
        batches, d0, d1 = dataset_generation(N, problem_type="test")
    for X in batches:

        f, phi_faces, alpha = X
        f = f.to(device)
        phi_faces = phi_faces.to(device)
        model.f = f.clone().detach().requires_grad_(True)
        model.phi_faces = phi_faces.clone().detach().requires_grad_(True)
        u_it = model.forward_eval(f)
        error = perturb.forward(f)
        u_est = u_it + epsilon * error.squeeze(-1).squeeze(-1)

        phi_faces = phi_faces.view(N,N).t()
        u_est = u_est.view(N,N).t()
        phi_faces = phi_faces.cpu()
        u_est = u_est.cpu()

        #plot_results(phi_faces, u_est, N, problem_type,alpha,show_plot=show_plot)
        plot_contour(u_est, phi_faces)
    np.savez('../results/losses.npz', losses=losses)

    return losses  