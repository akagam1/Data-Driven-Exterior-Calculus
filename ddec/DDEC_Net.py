import torch
import torch.nn as nn
import torch.nn.init as init

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
        self.GRAD_s = None
        self.DIV = None
        self.f = None

        if self.epsilon > 0:
            hidden_dim = 5 
            self.nn_model = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            ).to(dtype=torch.float64)
            self._init_nn()

        self.d1 = properties['d1']
        self.d0 = properties['d0']
        self.device = self.d1.device

        self.B1_vals = nn.Parameter(torch.randn(self.d1.shape[1]))
        self.B2_vals = nn.Parameter(torch.randn(self.d1.shape[0]))
        self.D1_vals = nn.Parameter(torch.randn(self.d1.shape[1]))
        self.D2_vals = nn.Parameter(torch.randn(self.d1.shape[0]))
    
    def _init_nn(self):
        for layer in self.nn_model:
            if isinstance(layer, nn.Linear):
                if layer.out_features == self.out_dim:
                    init.zeros_(layer.weight)
                    init.zeros_(layer.bias)
                else:
                    init.kaiming_normal_(layer.weight, nonlinearity='tanh')
                    if layer.bias is not None:
                        init.zeros_(layer.bias)

    
    def compute_hodge_laplacian(self):
        B1 = torch.diag(self.B1_vals**2 + 1e-5 * torch.ones_like(self.B1_vals)).to(dtype=torch.float64)
        B2 = torch.diag(self.B2_vals**2).to(dtype=torch.float64)  
        D1 = torch.diag(self.D1_vals**2 + 1e-5 * torch.ones_like(self.D1_vals)).to(dtype=torch.float64)
        D2 = torch.diag(self.D2_vals**2).to(dtype=torch.float64) 
        B1_inv = torch.inverse(B1)
        D1_inv = torch.inverse(D1)

        self.DIV = B2 @ self.d1 @ B1_inv
        self.GRAD_s = D1_inv @ self.d1.T @ D2
        K = self.DIV@self.GRAD_s
        K = K + 1e-6*torch.eye(K.shape[0]).to(dtype=torch.float64)

        return K


    def forward(self, u, f):
        K = self.compute_hodge_laplacian()
        self.f = f
        u_new = self.forward_problem(u, K, f)
        return u_new
    
    def _transpose(self, x):
        return x.permute(*torch.arange(x.ndim - 1, -1, -1))
    
    def forward_problem(self, u, K, f):
        def operator(u):
            grad_u = self.GRAD_s @ u
            nn_val = self.nn_model(grad_u.unsqueeze(-1)).squeeze(-1)
            div_nn_val = self.DIV @ nn_val
            Ku = K@u + self.epsilon * div_nn_val

            return Ku - f
        
        u_n = u.clone().detach().requires_grad_(True)

        #K_bc, f_bc = self.apply_bcs(K, f)      
        u_n = torch.linalg.solve(K, f)
        for i in range(self.iter):
            if self.epsilon > 0:
                Ku = operator(u_n)
                residual_vec = Ku - f
                J = torch.autograd.functional.jacobian(operator, u_n)
            
            else:
                Ku = K @ u_n
                residual_vec = K @ u_n - f
                J = K
            
            residual_norm = torch.linalg.norm(residual_vec, ord=2).item()
            if residual_norm < self.tol:
                break
            du = torch.linalg.solve(J, residual_vec)
            u_n = u_n - du

        if self.epsilon > 0:
            Ku = operator(u_n)
            J = torch.autograd.functional.jacobian(operator, u_n)
        else:
            Ku = K @ u_n
            J = K

        self.lambda_adj = self.adj_problem(u_n, J)
        self.adj_loss = self._transpose(self.lambda_adj) @ (Ku - f)

        return u_n

    def adj_problem(self, u_new, J):
        lambda_adj = torch.linalg.solve(J.T, 2*(self.phi_faces - u_new))
        return lambda_adj

