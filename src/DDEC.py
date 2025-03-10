"""
1) Need functions to return k-chains, and k-cochains
2) Accordingly need to define the boundary and coboundary operators on these
3) Add GRAD, CURL, DIV (and GRAD*, CURL*, DIV*) definitions as parameters to the model
4) Define the Hodge Laplacian in terms of these
5) HL.u + NN() = f
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import generate_darcy as darcy
import torch

def get_faces(graph, N):
    """
    idea: for every quadrilateral face (rows) we have a column for each edge. 1 if its part of that particular face.
          -1 for the opposite direction (cw and ccw)
    """
    faces = []
    for i in range(1,N):
        for j in range(0,N-1):
            v1 = (i,j)
            v2 = (i,j+1)
            v3 = (i+1,j+1)
            v4 = (i+1,j)
            face = np.zeros(N*N)
            face[i*N + j] = 1 
            face[i*N+j+1] = 1 
            face[(i-1)*N + j+1] = 1 
            face[(i-1)*N + j] = 1
            faces.append(face)
            face = -1*face 
            faces.append(face)

    return np.array(faces)

def cobound_d1(graph):
    return torch.tensor(get_faces(graph))

def cobound_d0(graph):
    b_plus = nx.incidence_matrix(graph, oriented=True).toarray().T 
    b_minus = -1*b_plus
    b = torch.concatenate((torch.tensor(b_plus), torch.tensor(b_minus)), axis = 0)

    return b



N = 10
alpha = 1
a = 0.2

G, phi_grid, pos = darcy.solve_darcy_flow(N=N, alpha=alpha, a=a)
print(nx.algorithms.planarity.is_planar(G))

B = cobound_d0(G)
F = get_faces(G,N)
print(F.shape)
#print(F.shape)


nodes = list(G.nodes())
print(len(nodes))
print(B.shape)


