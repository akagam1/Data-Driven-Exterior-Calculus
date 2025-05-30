import numpy as np
import networkx as nx
import torch


def get_faces(graph):
    """Extract faces from a 2D grid graph with nodes as (x, y) tuples
    Args:
        graph (nx.Graph): A NetworkX graph representing a 2D grid with nodes as (x, y) tuples.

    Returns:
        list: A list of faces, where each face is represented as a list of edges in counter-clockwise order.
    """

    nodes = list(graph.nodes())
    if not nodes or not all(isinstance(node, tuple) and len(node) == 2 for node in nodes):
        return []
    xs = sorted({x for x, y in nodes})
    ys = sorted({y for x, y in nodes})
    faces = []

    for i in range(len(xs)-1):
        for j in range(len(ys)-1):
            # Define potential face vertices
            v00 = (xs[i], ys[j])
            v01 = (xs[i], ys[j+1])
            v11 = (xs[i+1], ys[j+1])
            v10 = (xs[i+1], ys[j])

            if (graph.has_node(v00) and graph.has_node(v01) and
                graph.has_node(v11) and graph.has_node(v10) and
                graph.has_edge(v00, v01) and graph.has_edge(v01, v11) and
                graph.has_edge(v11, v10) and graph.has_edge(v10, v00)):
                
                # Store edges in counter-clockwise order
                face_edges = [
                    (v00, v01),
                    (v01, v11),
                    (v11, v10),
                    (v10, v00)
                ]
                faces.append(face_edges)
                
    return faces

def cobound_d1(graph):
    """Create d1 coboundary operator (curl) with proper orientation handling
    Args:
        graph (nx.Graph): A NetworkX graph representing a 2D grid with nodes as (x, y) tuples.
    
    Returns:
        torch.Tensor: A 2D tensor representing the d1 coboundary operator.
    """
    edges = list(graph.edges())
    edge_to_index = {tuple(sorted(edge)): i for i, edge in enumerate(edges)}
    faces = get_faces(graph)

    num_faces = len(faces)
    num_edges = len(edges)
    
    if num_faces == 0:
        return torch.zeros((0, num_edges), dtype=torch.float64)

    d1 = np.zeros((num_faces, num_edges))

    for face_idx, face_edges in enumerate(faces):
        for u, v in face_edges:
            # Get canonical edge representation
            sorted_edge = tuple(sorted((u, v)))
            
            # Check if edge exists in graph
            if sorted_edge in edge_to_index:
                edge_idx = edge_to_index[sorted_edge]
                # Determine sign based on edge direction
                sign = 1 if (u, v) == sorted_edge else -1
                d1[face_idx, edge_idx] = sign

    return torch.tensor(d1, dtype=torch.float64)



def cobound_d0(graph):
    """Create d0 coboundary operator (divergence) for a 2D grid graph
    Args:
        graph (nx.Graph): A NetworkX graph representing a 2D grid with nodes as (x, y) tuples.
    Returns:
        torch.Tensor: A 2D tensor representing the d0 coboundary operator.
    """

    b = nx.incidence_matrix(graph, oriented=True).toarray().T 
    return torch.tensor(b, dtype=torch.float64)

def identify_face_bcs(G, problem_type='D1'):
    """Identify boundary faces for a 2D grid graph
    Args:
        G (nx.Graph): A NetworkX graph representing a 2D grid with nodes as (x, y) tuples.
        problem_type (str): Type of problem, either 'D1' or 'D2'.
    Returns:
        list: A list of indices of boundary faces.
    """
    faces = get_faces(G)

    boundary_faces = []
    for face_idx, face_edges in enumerate(faces):
        for u, v in face_edges:
            if (u[0] == 0 or v[0] == 0) and problem_type == 'D1':
                boundary_faces.append(face_idx)
                break
                
    return boundary_faces

def apply_face_bcs(f, boundary_faces, problem_type='D1', alpha=1.0):
    """Apply boundary conditions to the face cochain
    Args:
        f (torch.Tensor): The face cochain vector.
        boundary_faces (list): List of indices of boundary faces.
        problem_type (str): Type of problem, either 'D1' or 'D2'.
        alpha (float): Value to set for the boundary conditions.
    Returns:
        torch.Tensor: The modified face cochain vector with boundary conditions applied.
        list: List of tuples containing the index and value of each boundary condition.
    """
    f_bc = f.clone()
    bcs = []
    for idx in boundary_faces:
        f_bc[idx] = 1.0 if problem_type == 'D1' else alpha
        bcs.append((idx, f_bc[idx]))
    return f_bc, bcs

def convert_cochain(phi,N, degree=2):
    """Convert a cochain vector to face cochain vector based on the degree
    Args:
        phi (torch.Tensor): The cochain vector.
        N (int): Size of the grid.
        degree (int): Degree of the cochain, either 1 or 2.
    Returns:
        torch.Tensor: The face cochain vector.
    """
    
    if degree == 2:
        phi_faces = torch.zeros((N-1)*(N-1), dtype=torch.float64)
        for i in range(N-1):
            for j in range(N-1):
                phi_faces[i*(N-1)+j] = (phi[i*N+j] + phi[i*N+j+1] + phi[(i+1)*N+j+1] + phi[(i+1)*N+j]) / 4

    else:
        raise ValueError("Degree not supported")
    return phi_faces

