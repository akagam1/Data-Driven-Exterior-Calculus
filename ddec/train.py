import matplotlib.pyplot as plt
import numpy as np
from DDEC_Net import DDECModel, PerturbNet
import torch
from tqdm import tqdm
from utils import *


def train_main(N, alpha, iter, tol, epsilon, in_dim, out_dim, epochs, problem_type, lr,show_plot=False):

    batches, d0, d1 = dataset_generation(N, problem_type=problem_type)
    d0 = d0.to(device='cuda' if torch.cuda.is_available() else 'cpu')
    d1 = d1.to(device='cuda' if torch.cuda.is_available() else 'cpu')
    properties = {'d0': d0, 'd1': d1}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DDECModel(iter, tol, 0, in_dim, out_dim, properties)

    perturb = PerturbNet((N-1)*(N-1), 20, device=device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(perturb.parameters()), lr=lr)

    criterion = torch.nn.MSELoss()

    losses = []
    epochs = epochs

    u = make_u(N)

    print(f"Using device: {device}")
    model = model.to(device)
    u = u.to(device)
    perturb = perturb.to(device)

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

    print(f"Training completed after {epochs} epochs with final loss: {loss.item()}")

    for X in batches:

        f, phi_faces, alpha = X
        f = f.to(device)
        phi_faces = phi_faces.to(device)
        model.f = f.clone().detach().requires_grad_(True)
        model.phi_faces = phi_faces.clone().detach().requires_grad_(True)
        u_it = model.forward(u, f)
        error = perturb.forward(f)
        u_est = u_it + epsilon * error.squeeze(-1).squeeze(-1)
        phi_faces = phi_faces.cpu()
        u_est = u_est.cpu()

        plot_results(phi_faces, u_est, N, problem_type,alpha,show_plot=show_plot)
    np.savez('../results/losses.npz', losses=losses)


    return losses
    