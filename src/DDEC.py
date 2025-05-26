import numpy as np
import networkx as nx
import torch


def get_faces(graph):
    nodes = list(graph.nodes())
    if not nodes or not all(isinstance(node, tuple) and len(node) == 2 for node in nodes):
        return []
    xs = sorted({x for x, y in nodes})
    ys = sorted({y for x, y in nodes})
    faces = []

    # Detect quadrilateral faces
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
    """Create d1 coboundary operator (curl) with proper orientation handling"""
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
    b = nx.incidence_matrix(graph, oriented=True).toarray().T 
    return torch.tensor(b, dtype=torch.float64)

def identify_face_bcs(G, problem_type='D1'):
    """Identify boundary faces and their indices in f_n"""
    faces = get_faces(G)

    boundary_faces = []
    for face_idx, face_edges in enumerate(faces):
        for u, v in face_edges:
            if (u[0] == 0 or v[0] == 0) and problem_type == 'D1':
                boundary_faces.append(face_idx)
                break  # Mark entire face as boundary
                
    return boundary_faces

def apply_face_bcs(f, boundary_faces, problem_type='D1', alpha=1.0):
    """Apply BCs directly to face-based f_n vector"""
    f_bc = f.clone()
    bcs = []
    for idx in boundary_faces:
        f_bc[idx] = 1.0 if problem_type == 'D1' else alpha
        bcs.append((idx, f_bc[idx]))
    return f_bc, bcs

def convert_cochain(phi,N, degree=2):
    """Converts a 0-cochain to a 2-cochain
        Need to generalize the functtonality eventually for 0-cochain to d-cochain
        phi: 0-cochain vector flattened out column wise
        constructing the 2-cochain colum wise too
 """
    if degree == 2:
        phi_faces = torch.zeros((N-1)*(N-1), dtype=torch.float64)
        for i in range(N-1):
            for j in range(N-1):
                phi_faces[i*(N-1)+j] = (phi[i*N+j] + phi[i*N+j+1] + phi[(i+1)*N+j+1] + phi[(i+1)*N+j]) / 4

    else:
        raise ValueError("Degree not supported")
    return phi_faces

