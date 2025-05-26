import torch
import torch.nn as nn

class DDECModel(nn.Module):
    def __init__(self, iter, tol, epsilon, in_dim, out_dim, properties):
        super(DDECModel, self).__init__()
        self.iter = iter
        self.tol = tol
        self.epsilon = epsilon
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lambda_adj = 0
        self.values = []
        self.adj_loss = 0
        self.bcs = []
        self.phi_faces = []

        # if (self.epsilon > 0):
        #     #neural net setup
        #     continue

        self.d1 = properties['d1']
        self.d0 = properties['d0']
        self.f = properties['f']
        self.device = self.d1.device

        self.B1_vals = nn.Parameter(torch.randn(self.d1.shape[1]))
        self.B2_vals = nn.Parameter(torch.randn(self.d1.shape[0]))
        self.D1_vals = nn.Parameter(torch.randn(self.d1.shape[1]))
        self.D2_vals = nn.Parameter(torch.randn(self.d1.shape[0]))
    
    def apply_bcs(self, J, rhs):
        J_bc = J.clone()
        rhs_bc = rhs.clone()
        
        for bc in self.bcs:
            idx, val = bc
            J_bc[idx, :] = 0
            J_bc[idx, idx] = 1
            rhs_bc[idx] = val

        return J_bc, rhs_bc
    
    def compute_hodge_laplacian(self):
        B1 = torch.diag(self.B1_vals**2 + 1e-5 * torch.ones_like(self.B1_vals)).to(dtype=torch.float64)
        B2 = torch.diag(self.B2_vals**2).to(dtype=torch.float64)  
        D1 = torch.diag(self.D1_vals**2 + 1e-5 * torch.ones_like(self.D1_vals)).to(dtype=torch.float64)
        D2 = torch.diag(self.D2_vals**2).to(dtype=torch.float64) 
        B1_inv = torch.inverse(B1)
        D1_inv = torch.inverse(D1)

        DIV = B2 @ self.d1 @ B1_inv
        GRAD_s = D1_inv @ self.d1.T @ D2
        K = DIV@GRAD_s
        K = K + 1e-6*torch.eye(K.shape[0]).to(dtype=torch.float64)

        return K


    def forward(self, u, f):
        K = self.compute_hodge_laplacian()
        u_new = self.forward_problem(u, K, f)
        return u_new

    def forward_problem(self, u, K, f):
        u_n = u.clone().detach().requires_grad_(True)
        
        iteration = 1

        #K_bc, f_bc = self.apply_bcs(K, f)
    
        if (iteration == 1):
            u_n = torch.linalg.solve(K, f)
            iteration += 1
        if (iteration > 1):
            for i in range(self.iter):
                residual_vec = K @ u_n - f
                residual_norm = torch.linalg.norm(residual_vec, ord=2).item()
                if residual_norm < self.tol:
                    break
                du = torch.linalg.solve(K, residual_vec)
                u_n = u_n - du
        self.lambda_adj = self.adj_problem(u_n, K)
        self.adj_loss = self.lambda_adj.T @ (K @ u_n - f)

        return u_n

    def adj_problem(self, u_new, K):
        lambda_adj = torch.linalg.solve(K.T, 2*(self.phi_faces - u_new))
        return lambda_adj

