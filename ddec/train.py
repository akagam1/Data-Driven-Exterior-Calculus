import matplotlib.pyplot as plt
from DDEC_Net import DDECModel
import torch
from tqdm import tqdm
from utils import *


def train_main(N, alpha, iter, tol, epsilon, in_dim, out_dim, epochs, problem_type, lr,show_plot=False):

    batches, d0, d1 = dataset_generation(N, problem_type=problem_type)
    d0 = d0.to(device='cuda' if torch.cuda.is_available() else 'cpu')
    d1 = d1.to(device='cuda' if torch.cuda.is_available() else 'cpu')
    properties = {'d0': d0, 'd1': d1}

    model = DDECModel(iter, tol, epsilon, in_dim, out_dim, properties)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = torch.nn.MSELoss()

    losses = []
    epochs = epochs

    u = make_u(N)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    u = u.to(device)


    with tqdm(total= epochs, desc="Training", unit="epoch", colour='green') as pbar:
        for epoch in range(epochs):
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

                    l = model.adj_loss

                    loss = criterion(u_it, phi_faces) + l
                    loss.backward(retain_graph=True)


                optimizer.step()
                losses.append(loss.item())
            val = model.get_nn_contrib()

            pbar.set_postfix({
        "loss": f"{loss.item():.4f}",
        "NN val ": f"{val:.4e}"
    })
            pbar.update(1)

            # if (loss.item() < 1e-10):
            #     print(f"Converged at epoch {epoch} with loss {loss.item()}")
            #     brea
    for X in batches:

        f, phi_faces, alpha = X
        model.f = f.clone().detach().requires_grad_(True)
        model.phi_faces = phi_faces.clone().detach().requires_grad_(True)
        u_est = model.forward(u, f)

        plot_results(phi_faces, u_est, N, problem_type,alpha,show_plot=show_plot)
    np.savez('../results/losses.npz', losses=losses)


    return losses
    