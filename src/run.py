import argparse
import sys
import os
from train import train_main

parser = argparse.ArgumentParser(description="Run the PDE constrained optimization for specific problem types.")

parser.add_argument('--N', type=int, default=10, help='Grid size for the problem (default: 10)')
parser.add_argument('--alpha', type=float, default=1.0, help='Boundary condition value (default: 1.0)')
parser.add_argument('--iter', type=int, default=20000, help='Number of iterations for the optimization (default: 1000)')
parser.add_argument('--tol', type=float, default=1e-12, help='Tolerance for convergence (default: 1e-5)')
parser.add_argument('--epsilon', type=float, default=0.0, help='Regularization parameter (default: 0.0)')
parser.add_argument('--in_dim', type=int, default=0, help='Input dimension for the model (default: 0)')
parser.add_argument('--out_dim', type=int, default=0, help='Output dimension for the model (default: 0)')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs for training (default: 10000)')
parser.add_argument('--problem_type', type=str, choices=['D1', 'D2'], default='D1', help='Type of problem to solve (default: D1)')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for the optimizer (default: 0.005)')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
args = parser.parse_args()

if args.verbose:
    print("=" * 50)
    print("Running with the following parameters:")
    print(f"N: {args.N}")
    print(f"alpha: {args.alpha}")
    print(f"iter: {args.iter}")
    print(f"tol: {args.tol}")
    print(f"epsilon: {args.epsilon}")
    print(f"in_dim: {args.in_dim}")
    print(f"out_dim: {args.out_dim}")
    print(f"epochs: {args.epochs}")
    print(f"problem_type: {args.problem_type}")   
    print(f"lr: {args.lr}")
    print("=" * 50)

train_main(
        N=args.N,
        alpha=args.alpha,
        iter=args.iter,
        tol=args.tol,
        epsilon=args.epsilon,
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        epochs=args.epochs,
        problem_type=args.problem_type,
        lr=args.lr
    )