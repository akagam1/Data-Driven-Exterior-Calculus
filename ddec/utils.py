import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import generate_darcy_mod as darcy_mod
import DDEC as ddec
import torch
import matplotlib.pyplot as plt

def plot_solution(model_out, N, alpha, a):
    plt.figure(figsize=(10, 8))

    plt.title(f"Darcy Flow Solution (α={alpha}, a={a})")
    
    x = np.linspace(-0.5, 0.5, N)
    y = np.linspace(-0.5, 0.5, N)
    X, Y = np.meshgrid(x, y)
    
    contour = plt.contourf(X, Y, model_out.T, 20, cmap='viridis')
    plt.colorbar(contour, label='Pressure φ')
    
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
        pos = G.nodes[node]['pos']
        if abs(pos[0]) < 1e-10:
            f[i] = 1 if problem_type == 'D1' else alpha
        elif abs(pos[0] - 1.0) < 1e-10:
            f[i] = 0
    return f

def dataset_generation(N, problem_type='D1',alpha=1.0):
    batches = []
    if problem_type == 'D1':
        G = darcy_mod.create_darcy_dataset(n=N, problem_type=problem_type)
        node_phi = nx.get_node_attributes(G, 'phi')
        phi = torch.tensor(list(node_phi.values()), dtype=torch.float64)


        d0 = ddec.cobound_d0(G)
        d1 = ddec.cobound_d1(G)
        phi_faces = ddec.convert_cochain(phi, N, degree=2).clone().detach().requires_grad_(True)

        f = torch.zeros((N*N,), dtype=torch.float64)
        f_n = set_boundary_conditions(G, f, problem_type=problem_type, alpha=alpha)
        f_n = ddec.convert_cochain(f_n, N, degree=2)

        f = f_n.clone().detach().requires_grad_(True)
        batches = [(f, phi_faces,alpha)]

    if problem_type == 'D2':
        for alph in [1,2,4]:
            G = darcy_mod.create_darcy_dataset(n=N, problem_type=problem_type, alphas=[alph])
            node_phi = nx.get_node_attributes(G, 'phi')
            phi = torch.tensor(list(node_phi.values()), dtype=torch.float64)

            d0 = ddec.cobound_d0(G)
            d1 = ddec.cobound_d1(G)
            phi_faces = ddec.convert_cochain(phi, N, degree=2).clone().detach().requires_grad_(True)

            f = torch.zeros((N*N,), dtype=torch.float64)
            f_n = set_boundary_conditions(G, f, problem_type=problem_type, alpha=alph)
            f_n = ddec.convert_cochain(f_n, N, degree=2)

            f = f_n.clone().detach().requires_grad_(True)
            batches.append((f, phi_faces,alph))
    
    if problem_type == 'test':
        for alph in [3,2.5,5]:
            G = darcy_mod.create_darcy_dataset(n=N, problem_type=problem_type, alphas=[alph])
            node_phi = nx.get_node_attributes(G, 'phi')
            phi = torch.tensor(list(node_phi.values()), dtype=torch.float64)

            d0 = ddec.cobound_d0(G)
            d1 = ddec.cobound_d1(G)
            phi_faces = ddec.convert_cochain(phi, N, degree=2).clone().detach().requires_grad_(True)

            f = torch.zeros((N*N,), dtype=torch.float64)
            f_n = set_boundary_conditions(G, f, problem_type=problem_type, alpha=alph)
            f_n = ddec.convert_cochain(f_n, N, degree=2)

            f = f_n.clone().detach().requires_grad_(True)
            batches.append((f, phi_faces,alph))
    
    return batches, d0, d1

def make_u(N):
    u = abs(torch.randn((N-1)*(N-1), dtype=torch.float64, requires_grad=True))
    for i in range(u.shape[0]):
        if (u[i] >= 0):
            u[i] = min(0.75, u[i])
        else:
            u[i] = max(0, u[i])
    return u

def plot_results(phi_faces,u_est, N, problem_type,alpha,show_plot=False):
    plt.figure(figsize=(10, 6))
    plt.plot(phi_faces.detach().numpy(), label="phi_faces", linestyle='-', marker='o')
    plt.plot(u_est.detach().numpy(), label="u_est", linestyle='--', marker='s')

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Comparison of phi_faces and u_est")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    if problem_type == 'D1':
        plt.savefig(f'../results/N={N}_D1_results.png')
    elif problem_type == 'D2':
        plt.savefig(f'../results/N={N}_alpha={alpha}_D2_results.png')
    if show_plot:
        plt.show()
    print(f"Output plot saved in /results")
    #plt.show()