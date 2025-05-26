import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

def create_grid_graph(n):
    G = nx.grid_2d_graph(n, n)
    pos = {}
    for i in range(n):
        for j in range(n):
            pos[(i, j)] = (i/(n-1), j/(n-1))
    nx.set_node_attributes(G, pos, 'pos')
    return G

def compute_diffusion_coefficient(G, alpha, radius):
    mu = {}
    center = (0.5, 0.5)
    for node in G.nodes():
        pos = G.nodes[node]['pos']
        distance = np.sqrt((pos[0] - center[0])**2 + (pos[1] - center[1])**2)
        mu[node] = alpha if distance < radius else 1.0
    nx.set_node_attributes(G, mu, 'mu')
    return G

def setup_darcy_problem(G, problem_type='D1', alpha=2.0):
    n = int(np.sqrt(G.number_of_nodes()))
    N = G.number_of_nodes()
    A = lil_matrix((N, N))
    b = np.zeros(N)
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    for node in G.nodes():
        i = node_to_idx[node]
        mu_i = G.nodes[node]['mu']
        diag_term = 0
        for neighbor in G.neighbors(node):
            j = node_to_idx[neighbor]
            mu_j = G.nodes[neighbor]['mu']
            mu_ij = 2 * mu_i * mu_j / (mu_i + mu_j)
            A[i, j] = -mu_ij
            diag_term += mu_ij
        A[i, i] = diag_term
    for node in G.nodes():
        i = node_to_idx[node]
        pos = G.nodes[node]['pos']
        if abs(pos[0]) < 1e-10:
            A[i, :] = 0
            A[i, i] = 1
            b[i] = 1 if problem_type == 'D1' else alpha
        elif abs(pos[0] - 1.0) < 1e-10:
            A[i, :] = 0
            A[i, i] = 1
            b[i] = 0
    return A.tocsr(), b, node_to_idx

def solve_darcy_problem(G, problem_type='D1', alpha=2.0, radius=0.25):
    G = compute_diffusion_coefficient(G, alpha, radius)
    A, b, node_to_idx = setup_darcy_problem(G, problem_type, alpha)
    phi = spsolve(A, b)
    for node in G.nodes():
        i = node_to_idx[node]
        G.nodes[node]['phi'] = phi[i]
    for u, v in G.edges():
        mu_u = G.nodes[u]['mu']
        mu_v = G.nodes[v]['mu']
        mu_uv = 2 * mu_u * mu_v / (mu_u + mu_v)
        phi_u = G.nodes[u]['phi']
        phi_v = G.nodes[v]['phi']
        flux = -mu_uv * (phi_v - phi_u)
        G[u][v]['flux'] = flux
    return G

def plot_solution_with_circle(G, title, radius=0.25):
    plt.figure(figsize=(10, 8))
    node_pos = nx.get_node_attributes(G, 'pos')
    node_phi = nx.get_node_attributes(G, 'phi')
    x = [pos[0] for pos in node_pos.values()]
    y = [pos[1] for pos in node_pos.values()]
    phi = list(node_phi.values())
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

def create_darcy_dataset(n=50, problem_type='D1', alphas=None):
    if alphas is None:
        alphas = [1.0] if problem_type == 'D1' else [1.0, 2.0, 4.0]
    for alpha in alphas:
        G = create_grid_graph(n)
        G = solve_darcy_problem(G, problem_type, alpha)
        # plot_solution_with_circle(G, f"{problem_type} with α={alpha}")
    return G


# Example usage
# n=6
# create_darcy_dataset(n=n, problem_type='D1')
# # create_darcy_dataset(n=50, problem_type='D2')
