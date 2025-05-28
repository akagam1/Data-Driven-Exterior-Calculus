import matplotlib.pyplot as plt
from DDEC_Net import DDECModel
import torch
from tqdm import tqdm
from utils import *

def train_main(N, alpha, iter, tol, epsilon, in_dim, out_dim, epochs, problem_type, lr):

    batches, d0, d1 = dataset_generation(N, problem_type=problem_type)

    properties = {'d0': d0, 'd1': d1}

    model = DDECModel(iter, tol, epsilon, in_dim, out_dim, properties)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = torch.nn.MSELoss()

    losses = []
    epochs = epochs

    u = make_u(N)

    with tqdm(total= epochs, desc="Training", unit="epoch") as pbar:
        for epoch in range(epochs):
            for X in batches:
                f, phi_faces = X
                model.phi_faces = phi_faces.clone().detach().requires_grad_(True)
                optimizer.zero_grad()
                with torch.enable_grad():
                    u_det = u.detach().requires_grad_(True)
                    u_it = model(u_det, f)

                    l = model.adj_loss

                    loss = criterion(u_it, phi_faces) + l
                    loss.backward(retain_graph=True)

                optimizer.step()
                losses.append(loss.item())

            pbar.set_postfix({'Loss': f'{loss.item():.10f}'})
            pbar.update(1)

            if (loss.item() < 1e-10):
                print(f"Converged at epoch {epoch} with loss {loss.item()}")
                break

    if (len(batches) > 1):
        f,phi_faces = batches[2]
    u_est = model.forward(u, f) 

    plt.figure(figsize=(10, 6))
    plt.plot(phi_faces.detach().numpy(), label="phi_faces", linestyle='-', marker='o')
    plt.plot(u_est.detach().numpy(), label="u_est", linestyle='--', marker='s')

    # Labels and title
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Comparison of phi_faces and u_est")
    plt.legend()
    plt.grid()

    # Show plot
    plt.show()
    