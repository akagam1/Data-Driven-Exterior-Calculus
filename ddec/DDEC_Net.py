import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd.functional import jvp

class PerturbNet(nn.Module):
    def __init__(self, N, hidden_dim=5, device='cuda'):
        super(PerturbNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.nn_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 1)
            ).to(dtype=torch.float64, device=self.device) for _ in range(N)
        ])
        
        self._init_nn()
    
    def _init_nn(self):
        """
        Initializes the neural network weights and biases using He initialization.
        """
        for model in self.nn_modules:
            for layer in model:
                if isinstance(layer, nn.Linear):
                    init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    if layer.bias is not None:
                        init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        Forward pass of the neural network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 1).
        
        Returns:
            torch.Tensor: Output tensor of shape (N, 1).
        """
        outputs = []
        #x is of shape (num_modules, 1)
        for i, module in enumerate(self.nn_modules):
            output = module(x[i].unsqueeze(0).unsqueeze(-1)).squeeze(-1)
            outputs.append(output)
        return torch.stack(outputs, dim=0).to(dtype=torch.float64, device=self.device)


class DDECModel(nn.Module):
    def __init__(self, iter, tol, in_dim, out_dim, properties, cochain=2, device='cuda'):
        super(DDECModel, self).__init__()
        self.device = device
        if device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
        self.cochain = cochain
        self.iter = iter
        self.tol = tol
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
        self.d1 = properties['d1']
        self.d0 = properties['d0']
        self.nn_contrib = 0
    
        self.device = self.d1.device
        self.B0_vals = nn.Parameter(torch.randn(self.d0.shape[1]))
        self.B1_vals = nn.Parameter(torch.randn(self.d1.shape[1]))
        self.B2_vals = nn.Parameter(torch.randn(self.d1.shape[0]))
        self.D0_vals = nn.Parameter(torch.randn(self.d0.shape[1]))
        self.D1_vals = nn.Parameter(torch.randn(self.d1.shape[1]))
        self.D2_vals = nn.Parameter(torch.randn(self.d1.shape[0]))

    
    def compute_hodge_laplacian(self):
        """
        Computes the Hodge Laplacian operator for the given d1 and d0 operators.
        
        Returns: 
            torch.Tensor: The Hodge Laplacian operator K.  
        """
        B0 = torch.diag(self.B0_vals**2 + 1e-5 * torch.ones_like(self.B0_vals)).to(dtype=torch.float64, device=self.device)
        B1 = torch.diag(self.B1_vals**2 + 1e-5 * torch.ones_like(self.B1_vals)).to(dtype=torch.float64, device=self.device)
        B2 = torch.diag(self.B2_vals**2).to(dtype=torch.float64, device=self.device)  
        D0 = torch.diag(self.D0_vals**2 + 1e-5 * torch.ones_like(self.D0_vals)).to(dtype=torch.float64, device=self.device)
        D1 = torch.diag(self.D1_vals**2 + 1e-5 * torch.ones_like(self.D1_vals)).to(dtype=torch.float64, device=self.device)
        D2 = torch.diag(self.D2_vals**2).to(dtype=torch.float64, device=self.device) 
        B0_inv = torch.inverse(B0)
        B1_inv = torch.inverse(B1)
        D0_inv = torch.inverse(D0)
        D1_inv = torch.inverse(D1)

        if self.cochain == 2:
            self.DIV = B2 @ self.d1 @ B1_inv
            self.GRAD_s = D1_inv @ self.d1.T @ D2
            K = self.DIV@self.GRAD_s
            K = K + 1e-6*torch.eye(K.shape[0]).to(dtype=torch.float64, device=self.device)
        if self.cochain == 0:
            CURL = B1 @ self.d0 @ B0_inv
            CURL_s = D0_inv @ self.d0.T @ D1
            K = CURL_s @CURL
            K = K + 5*torch.eye(K.shape[0]).to(dtype=torch.float64, device=self.device)
        return K


    def forward(self, u, f,epoch=101):
        self.epoch = epoch
        K = self.compute_hodge_laplacian()
        self.f = f
        u_new = self.forward_problem(u, K, f)
        return u_new
    
    def _transpose(self, x):
        return x.permute(*torch.arange(x.ndim - 1, -1, -1))
    
    
    def forward_problem(self, u, K, f):
        """
        Forward problem solver using Newton's method for the PDE
        write google style doc with type hints and description
        
        Args:
            u (torch.Tensor): Initial guess for the solution.
            K (torch.Tensor): Hodge Laplacian operator.
            f (torch.Tensor): Right-hand side vector of the PDE.
        
        Returns:
            torch.Tensor: The solution to the PDE.
        """
  
        u_n = u.clone().detach().requires_grad_(True)
        u_n = torch.linalg.solve(K, f)
        for i in range(self.iter):  
            Ku = K @ u_n
            residual_vec = Ku - f
            J = K
            du = torch.linalg.solve(J, residual_vec)
            
            residual_norm = torch.linalg.norm(residual_vec, ord=2).item()
            if residual_norm < self.tol:
                break
            u_n = u_n - du

        Ku = K @ u_n
        J = K

        self.lambda_adj = self.adj_problem(u_n, J)
        self.adj_loss = self._transpose(self.lambda_adj) @ (Ku - f)

        return u_n

    def adj_problem(self, u_new, J):
        """
        Adjoint problem solver to compute the Lagrange multiplier lambda_adj.

        Args:
            u_new (torch.Tensor): The solution to the forward problem.
            J (torch.Tensor): Jacobian matrix of the operator.
        
        Returns:
            torch.Tensor: The adjoint variable lambda_adj.
        """

        lambda_adj = torch.linalg.solve(J.T, 2*(self.phi_faces - u_new))
        return lambda_adj
    
    def forward_eval(self,f):
        """
        Forward pass for evaluation (not iterative)
        """
        K = self.compute_hodge_laplacian()
        return torch.linalg.solve(K,f)

