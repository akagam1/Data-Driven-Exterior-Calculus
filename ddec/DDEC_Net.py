import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd.functional import jvp

class DDECModel(nn.Module):
    def __init__(self, iter, tol, epsilon, in_dim, out_dim, properties, device='cuda'):
        super(DDECModel, self).__init__()
        self.device = device
        if device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
        
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
        self.d1 = properties['d1']
        self.d0 = properties['d0']
        self.nn_contrib = 0
        
        temp = self.d1 @ self.d0
        if self.epsilon > 0:
            hidden_dim = 20
            self._setup_neural_network(hidden_dim=hidden_dim)

        self.device = self.d1.device
        print(f"Using device: {self.device}")

        self.B1_vals = nn.Parameter(torch.randn(self.d1.shape[1]))
        self.B2_vals = nn.Parameter(torch.randn(self.d1.shape[0]))
        self.D1_vals = nn.Parameter(torch.randn(self.d1.shape[1]))
        self.D2_vals = nn.Parameter(torch.randn(self.d1.shape[0]))

    def _setup_neural_network(self, hidden_dim=20):
        """
        Sets up the neural network modules for the DDEC model.
        This method initializes the neural network layers and their parameters.
        """
        self.nn_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 1)
            ).to(dtype=torch.float64, device=self.device) for _ in range(self.d1.shape[1])
        ])
        #self._init_nn()
    
    def _init_nn(self):
        """
        Initializes the neural network weights and biases using He initialization.
        """
        for model in self.nn_modules:
            for layer in model:
                if isinstance(layer, nn.Linear):
                    if layer.out_features == self.out_dim:
                        init.zeros_(layer.weight)
                        init.zeros_(layer.bias)
                    else:
                        init.kaiming_normal_(layer.weight, nonlinearity='relu')
                        if layer.bias is not None:
                            init.zeros_(layer.bias)

    
    def compute_hodge_laplacian(self):
        """
        Computes the Hodge Laplacian operator for the given d1 and d0 operators.
        
        Returns: 
            torch.Tensor: The Hodge Laplacian operator K.  
        """
        B1 = torch.diag(self.B1_vals**2 + 1e-5 * torch.ones_like(self.B1_vals)).to(dtype=torch.float64, device=self.device)
        B2 = torch.diag(self.B2_vals**2).to(dtype=torch.float64, device=self.device)  
        D1 = torch.diag(self.D1_vals**2 + 1e-5 * torch.ones_like(self.D1_vals)).to(dtype=torch.float64, device=self.device)
        D2 = torch.diag(self.D2_vals**2).to(dtype=torch.float64, device=self.device) 
        B1_inv = torch.inverse(B1)
        D1_inv = torch.inverse(D1)

        self.DIV = B2 @ self.d1 @ B1_inv
        self.GRAD_s = D1_inv @ self.d1.T @ D2
        K = self.DIV@self.GRAD_s
        K = K + 1e-6*torch.eye(K.shape[0]).to(dtype=torch.float64, device=self.device) 
        return K

    def get_nn_contrib(self):
        """
        Returns the average absolute value of the neural network contributions.
        
        Returns:
            float: The average absolute value of the neural network contributions.
        """
        return self.nn_contrib


    def forward(self, u, f,epoch=101):
        self.epoch = epoch
        K = self.compute_hodge_laplacian()
        self.f = f
        u_new = self.forward_problem(u, K, f)
        return u_new
    
    def _transpose(self, x):
        return x.permute(*torch.arange(x.ndim - 1, -1, -1))
    
    def cg_solver(self, matvec, rhs, tol=1e-6, max_iter=100):
        """
        Conjugate Gradient solver for the linear system Ax = b.
        
        Args:
            matvec (callable): Function to compute the matrix-vector product Ax.
            rhs (torch.Tensor): Right-hand side vector b.
            tol (float): Tolerance for convergence.
            max_iter (int): Maximum number of iterations.
        
        Returns:
            torch.Tensor: Solution vector x.
        """

        x = torch.zeros_like(rhs, dtype=torch.float64)
        r = rhs.clone().detach().requires_grad_(True)
        p = r.clone()
        rsold = torch.dot(r, r)

        for i in range(max_iter):
            Ap = matvec(p)
            alpha = rsold / (torch.dot(p, Ap) + 1e-10)
            x += alpha * p
            r = r - alpha * Ap
            rsnew = torch.dot(r, r)

            if torch.sqrt(rsnew) < tol:
                break

            p = r + (rsnew / rsold) * p
            rsold = rsnew

        return x
    
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
        def matvec(v):
            """
            Matrix-vector product for the operator function
            Args:
                v (torch.Tensor): Input vector.
            Returns:
                torch.Tensor: Result of the matrix-vector product.
            """

            jvp_resullt = jvp(operator, (u_n,), (v,), create_graph=True)[1]
            return jvp_resullt

        def operator(u):
            grad_u = self.GRAD_s @ u
            #nn_val = self.nn_model(grad_u.unsqueeze(-1)).squeeze(-1)
            nn_val = torch.stack([self.nn_modules[i](grad_u[i].unsqueeze(0)).squeeze(0) for i in range(grad_u.shape[0])])
            with torch.no_grad():
                self.nn_contrib = nn_val.abs().mean().item()


            div_nn_val = self.DIV @ nn_val
            Ku = K@u + self.epsilon * div_nn_val

            return Ku
        
        u_n = u.clone().detach().requires_grad_(True)
    
        u_n = torch.linalg.solve(K, f)
        for i in range(self.iter):
            if self.epsilon > 0:
                Ku = operator(u_n)
                residual_vec = Ku - f
                J = torch.autograd.functional.jacobian(operator, u_n)
                #du = self.cg_solver(matvec, residual_vec, tol=1e-6)
            
            else:
                Ku = K @ u_n
                residual_vec = Ku - f
                J = K
            du = torch.linalg.solve(J, residual_vec)
            
            residual_norm = torch.linalg.norm(residual_vec, ord=2).item()
            if residual_norm < self.tol:
                break
            u_n = u_n - du

        if self.epsilon > 0:
            Ku = operator(u_n)
            #J = torch.autograd.functional.jacobian(operator, u_n)
        else:
            Ku = K @ u_n
            #J = K

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

