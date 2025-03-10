import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

def create_grid_graph(N):
    """Create a grid graph representing the discretization"""
    G = nx.grid_2d_graph(N, N)
    
    pos = {(i, j): ((i/(N-1) - 0.5), (j/(N-1) - 0.5)) for i, j in G.nodes()}
    nx.set_node_attributes(G, pos, 'pos')
    return G, pos

def compute_material_property(G, alpha, a):
    """Compute the material property mu_alpha at each node"""
    mu = {}
    for node in G.nodes():
        pos = G.nodes[node]['pos']
        x, y = pos
        r = np.sqrt(x**2 + y**2)
        mu[node] = alpha if r < a else 1.0
    nx.set_node_attributes(G, mu, 'mu')
    return G

def setup_system(G, N, alpha):
    """Set up the linear system to solve the PDE"""
    h = 1.0 / (N - 1)  
    n_nodes = N * N
    A = lil_matrix((n_nodes, n_nodes))
    b = np.zeros(n_nodes)
     
    node_to_idx = {node: i*N + j for i, j, node in zip(
        [n[0] for n in G.nodes()], 
        [n[1] for n in G.nodes()], 
        G.nodes()
    )}
    
    for node in G.nodes():
        i, j = node
        idx = node_to_idx[node]
        mu_node = G.nodes[node]['mu']
        
        # Interior nodes use standard 5-point stencil
        if 0 < i < N-1 and 0 < j < N-1:

            A[idx, idx] = -4 * mu_node / h**2
            
            east = (i+1, j)
            mu_e = 0.5 * (mu_node + G.nodes[east]['mu'])
            A[idx, node_to_idx[east]] = mu_e / h**2
            
            west = (i-1, j)
            mu_w = 0.5 * (mu_node + G.nodes[west]['mu'])
            A[idx, node_to_idx[west]] = mu_w / h**2
            
            north = (i, j+1)
            mu_n = 0.5 * (mu_node + G.nodes[north]['mu'])
            A[idx, node_to_idx[north]] = mu_n / h**2
            
            south = (i, j-1)
            mu_s = 0.5 * (mu_node + G.nodes[south]['mu'])
            A[idx, node_to_idx[south]] = mu_s / h**2
            
        # Boundary conditions
        elif i == N-1:  # Right (east) boundary - Neumann flux
            A[idx, idx] = -3 * mu_node / h**2
            
            west = (i-1, j)
            mu_w = 0.5 * (mu_node + G.nodes[west]['mu'])
            A[idx, node_to_idx[west]] = mu_w / h**2
            
            if j > 0:
                south = (i, j-1)
                mu_s = 0.5 * (mu_node + G.nodes[south]['mu'])
                A[idx, node_to_idx[south]] = mu_s / h**2
                
            if j < N-1:
                north = (i, j+1)
                mu_n = 0.5 * (mu_node + G.nodes[north]['mu'])
                A[idx, node_to_idx[north]] = mu_n / h**2
                
            # Add Neumann flux to RHS
            b[idx] = -alpha
            
        elif (i == 0 or j == 0 or j == N-1):  
            if i == 0 and j == 0:  # Fix phi(0,0) = 0 to remove singularity
                A[idx, idx] = 1
                b[idx] = 0
                continue
                
            # Set up boundary conditions with homogeneous Neumann

            A[idx, idx] = -3 * mu_node / h**2
            
            # Handle neighbors based on which boundary we're on
            neighbors = list(G.neighbors(node))
            for neighbor in neighbors:
                n_idx = node_to_idx[neighbor]
                mu_avg = 0.5 * (mu_node + G.nodes[neighbor]['mu'])
                A[idx, n_idx] = mu_avg / h**2 
    

    return A.tocsr(), b, node_to_idx

def solve_darcy_flow(N=101, alpha=0.1, a=0.25):
    """
    Solve the Darcy flow problem with a cylindrical inclusion
    
    Parameters:
    N : int, grid size (N x N)
    alpha : float, material property inside the cylinder
    a : float, radius of the cylinder
    
    Returns:
    G : networkx.Graph, graph with solution data
    phi : ndarray, solution values
    """
    # Create grid and set material properties
    G, pos = create_grid_graph(N)
    G = compute_material_property(G, alpha, a)
    
    # Set up and solve linear system
    A, b, node_to_idx = setup_system(G, N, alpha)

    phi = spsolve(A, b)
    phi+=0.55
    
    # Store solution on graph
    for node in G.nodes():
        idx = node_to_idx[node]
        G.nodes[node]['phi'] = phi[idx]
    
    return G, phi.reshape(N, N), pos

def plot_solution(G, phi_grid, pos, N, alpha, a):
    """Plot the solution"""
    plt.figure(figsize=(15, 6))
    
    # Plot the graph with node colors based on phi
    plt.subplot(1, 2, 1)
    plt.title(f"Darcy Flow Solution (α={alpha}, a={a})")
    
    # Extract x, y coordinates and phi values for contour plot
    x = np.linspace(-0.5, 0.5, N)
    y = np.linspace(-0.5, 0.5, N)
    X, Y = np.meshgrid(x, y)
    
    # Create contour plot
    contour = plt.contourf(X, Y, phi_grid.T, 20, cmap='viridis')
    plt.colorbar(contour, label='Pressure φ')
    
    # Draw the circle representing the inclusion
    circle = plt.Circle((0, 0), a, fill=False, color='r', linewidth=2)
    plt.gca().add_patch(circle)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    
    # Plot the graph
    plt.subplot(1, 2, 2)
    plt.title("Graph Representation")
    
    # Create a subset of nodes to visualize (plotting all nodes would be too dense)
    subset_indices = np.linspace(0, N-1, min(20, N), dtype=int)
    subset_nodes = [(i, j) for i in subset_indices for j in subset_indices]
    
    # Extract the subgraph for visualization
    subgraph = G.subgraph(subset_nodes)
    sub_pos = {node: pos[node] for node in subset_nodes}
    
    # Color nodes by phi value
    node_colors = [subgraph.nodes[node]['phi'] for node in subgraph.nodes()]
    
    # Draw the graph
    nx.draw_networkx(
        subgraph,
        pos=sub_pos,
        node_color=node_colors,
        cmap='viridis',
        node_size=50,
        with_labels=False,
        edge_color='gray',
        width=0.5
    )
    
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label='Pressure φ')
    plt.axis('equal')
    plt.tight_layout()
    
    return plt

def compute_velocity_field(G, phi_grid, N):
    """Compute the velocity field F = -mu_alpha * grad(phi)"""
    h = 1.0 / (N - 1)
    F_x = np.zeros((N, N))
    F_y = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            node = (i, j)
            if node in G.nodes():
                mu = G.nodes[node]['mu']
                
                # Compute gradient using central differences (with forward/backward at boundaries)
                if i == 0:
                    dx_phi = (phi_grid[i+1, j] - phi_grid[i, j]) / h
                elif i == N-1:
                    dx_phi = (phi_grid[i, j] - phi_grid[i-1, j]) / h
                else:
                    dx_phi = (phi_grid[i+1, j] - phi_grid[i-1, j]) / (2*h)
                    
                if j == 0:
                    dy_phi = (phi_grid[i, j+1] - phi_grid[i, j]) / h
                elif j == N-1:
                    dy_phi = (phi_grid[i, j] - phi_grid[i, j-1]) / h
                else:
                    dy_phi = (phi_grid[i, j+1] - phi_grid[i, j-1]) / (2*h)
                
                # F = -mu_alpha * grad(phi)
                F_x[i, j] = -mu * dx_phi
                F_y[i, j] = -mu * dy_phi
    
    return F_x, F_y

def plot_velocity_field(phi_grid, G, N, alpha, a):
    """Plot the pressure contours and velocity field"""
    F_x, F_y = compute_velocity_field(G, phi_grid, N)
    
    plt.figure(figsize=(10, 8))
    plt.title(f"Darcy Flow: Pressure and Velocity Field (α={alpha}, a={a})")
    
    # Extract x, y coordinates for plotting
    x = np.linspace(-0.5, 0.5, N)
    y = np.linspace(-0.5, 0.5, N)
    X, Y = np.meshgrid(x, y)
    
    # Create contour plot of pressure
    contour = plt.contourf(X, Y, phi_grid.T, 20, cmap='plasma', alpha=0.8)
    plt.colorbar(contour, label='Pressure φ')
    
    # Use a reduced set of points for clarity
    skip = max(1, N // 5)
    plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
               F_x[::skip, ::skip].T, F_y[::skip, ::skip].T,
               color='white', scale=20)
    
    # Draw the circle representing the inclusion
    circle = plt.Circle((0, 0), a, fill=False, color='r', linewidth=2)
    plt.gca().add_patch(circle)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.tight_layout()
    
    return plt

if __name__ == "__main__":
    # Parameter settings
    N = 20  # Grid size (N x N)
    alpha = 1  # Material property inside the cylinder
    a = 0.2  # Radius of the cylinder
    
    # Solve the PDE
    G, phi_grid, pos = solve_darcy_flow(N=N, alpha=alpha, a=a)
    print(phi_grid)
    
    # Plot solution
    plt_solution = plot_solution(G, phi_grid, pos, N, alpha, a)
    plt_solution.savefig('darcy_flow_solution.png', dpi=300)
    
    # Plot velocity field
    plt_velocity = plot_velocity_field(phi_grid, G, N, alpha, a)
    plt_velocity.savefig('darcy_flow_velocity.png', dpi=300)
    
    plt.show()
