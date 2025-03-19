class DDECNet(nn.Module):
    def __init__(self, iter, tol, epsilon, in_dim, out_dim, properties):
        super(DDECNet, self).__init__()
        self.iter = iter
        self.tol = tol
        self.epsilon = epsilon
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lambda_adj = 0

        if (self.epsilon > 0):
            #neural net setup
            continue

        #Setup Properties of the network
        self.d1 = properties['d1']
        self.d0 = properties['d0']
        self.f = properties['f']
        
        #set dimensions poperly
        self.B1_vals = nn.Parameter(torch.randn(n))
        self.B2_vals = nn.Parameter(torch.randn(n))
        self.D1_vals = nn.Parameter(torch.randn(n))
        self.D2_vals = nn.Parameter(torch.randn(n))

    def forward(self, u, f):
        B1 = torch.diag(self.B1_vals**2)
        B2 = torch.diag(self.B2_vals**2)
        D1 = torch.diag(self.D1_vals**2)
        D2 = torch.diag(self.D2_vals**2)

        B1_inv = torch.inverse(B1)
        D1_inv = torch.inverse(D1)

        DIV = B2 @ self.d1 @ B1_inv
        GRAD_s = D1_inv @ self.d1.T @ D2
        K = DIV@GRAD_s

        u_new = self.forward_problem(u, K, f)
        self.lambda_adj = self.adj_problem(u, u_new, K)

        return u

    def forward_problem(self, u, K, f):
        for i in range(self.iter):
            f_hat = K@u 
            residual = torch.linalg.norm(f_hat - f)
            if residual < self.tol:
                break

            J = K #jacbian is K itself (in linear case)
            du = torch.linalg.solve(J, f_hat - f)
            u += du

    def adj_problem(self, u, u_new, K):
        lambda_adj = torch.inverse(K).T @ (u - u_new)
        return lambda_adj

