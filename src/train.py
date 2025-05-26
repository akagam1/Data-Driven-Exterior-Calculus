import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import generate_darcy_mod as darcy_mod
import DDEC as ddec
from DDEC_Net import DDECModel
import torch
from tqdm import tqdm

def plot_solution(model_out, N, alpha, a):
    """Plot the solution"""
    plt.figure(figsize=(10, 8))

    plt.title(f"Darcy Flow Solution (α={alpha}, a={a})")
    
    x = np.linspace(-0.5, 0.5, N)
    y = np.linspace(-0.5, 0.5, N)
    X, Y = np.meshgrid(x, y)
    
    # Create contour plot
    contour = plt.contourf(X, Y, model_out.T, 20, cmap='viridis')
    plt.colorbar(contour, label='Pressure φ')
    
    # Draw the circle representing the inclusion
    circle = plt.Circle((0, 0), a, fill=False, color='r', linewidth=2)
    plt.gca().add_patch(circle)

    return plt

def plot_solution_with_circle(G, phi, title, radius=0.25):
    plt.figure(figsize=(10, 8))
    node_pos = nx.get_node_attributes(G, 'pos')
    x = [pos[0] for pos in node_pos.values()]
    y = [pos[1] for pos in node_pos.values()]
    n = int(np.sqrt(G.number_of_nodes()))
    X = np.reshape(x, (n, n))
    Y = np.reshape(y, (n, n))
    Z = np.reshape(phi, (n, n))
    plt.contourf(X, Y, Z, 20, cmap='viridis')
    plt.colorbar(label='Pressure')
    plt.title(f'Pressure Field and Flux Vectors - {title}')
    plt.xlabel('x')
    plt.ylabel('y')
    circle = plt.Circle((0.5, 0.5), radius, fill=False, color='r', linestyle='--')
    plt.gca().add_patch(circle)

    plt.tight_layout()
    plt.show()

def set_boundary_conditions(G, f, problem_type='D1', alpha=1.0):
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    for node in G.nodes():
        i = node_to_idx[node]
        #check if indexing is being done correctly
        pos = G.nodes[node]['pos']
        if abs(pos[0]) < 1e-10:
            f[i] = 1 if problem_type == 'D1' else alpha
        elif abs(pos[0] - 1.0) < 1e-10:
            f[i] = 0
    return f

def train_main(N, alpha, iter, tol, epsilon, in_dim, out_dim, epochs, problem_type, lr):
    #a = 0.2

    G = darcy_mod.create_darcy_dataset(n=N, problem_type=problem_type)
    node_phi = nx.get_node_attributes(G, 'phi')
    phi = torch.tensor(list(node_phi.values()), dtype=torch.float64)


    d0 = ddec.cobound_d0(G)
    d1 = ddec.cobound_d1(G)
    phi_faces = ddec.convert_cochain(phi, N, degree=2).clone().detach().requires_grad_(True)

    f = torch.zeros((N*N,), dtype=torch.float64)
    f_n = set_boundary_conditions(G, f, problem_type=problem_type, alpha=alpha)
    f_n = ddec.convert_cochain(f_n, N, degree=2)

    bcs = []
    # for i in range(f_n.shape[0]):
    #     if f_n[i] != 0:
    #         bcs.append((i, f_n[i]))
    f = f_n.clone().detach().requires_grad_(True)


    properties = {'d0': d0, 'd1': d1, 'f': f}

    model = DDECModel(iter, tol, epsilon, in_dim, out_dim, properties)
    model.bcs = bcs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = torch.nn.MSELoss()
    model.phi_faces = phi_faces


    losses = []
    epochs = 10000
    k = d1 @ d0
    def make_u(N):
        u = abs(torch.randn((N-1)*(N-1), dtype=torch.float64, requires_grad=True))
        for i in range(u.shape[0]):
            if (u[i] >= 0):
                u[i] = min(0.75, u[i])
            else:
                u[i] = max(0, u[i])
        return u

    u = make_u(N)

    with tqdm(total= epochs, desc="Training", unit="epoch") as pbar:
        for epoch in range(epochs):
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
    
#train_main(N=8, alpha=1, iter=20000, tol=1e-12, epsilon=0, in_dim=0, out_dim=0, epochs=10000)