import argparse
from train import train_main
import matplotlib.pyplot as plt

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
parser.add_argument('--loss-compare', action='store_true', help='Run loss comparison for different domain sizes')
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
    print(f"loss_compare: {args.loss_compare}")
    print("=" * 50)

if not args.loss_compare:
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
else:
    losses =[]

    if args.problem_type == 'D1':
        for N in [3,6,8]:
            loss = train_main(
                N=N,
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
            losses.append(loss)
        
        #plot loss comparison
        # 3 graphs for each N, iteration and loss
        plt.figure(figsize=(10, 6))
        for i, N in enumerate([3, 6, 8]):
            plt.plot(losses[i], label=f"N={N}", linestyle='-', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Comparison for Different Domain Sizes')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('loss_comparison.png')
        plt.show()
        print("Training completed successfully.")
        print("Loss comparison plot saved as 'loss_comparison.png'.")

    if args.problem_type == 'D2':
        loss = train_main(
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
        loss1=[]
        loss2=[]
        loss3=[]

        for i in range(args.epochs * 3):
            if int(i/3) == 0:
                loss1.append(loss[i])
            elif int(i/3) == 1:
                loss2.append(loss[i])
            elif int(i/3) == 2:
                loss3.append(loss[i])

        #plot loss comparison
        # 3 graphs for each alpha, iteration and loss

        plt.figure(figsize=(10, 6))
        plt.plot(loss1, label=f"alpha=1", linestyle='-', marker='o')
        plt.plot(loss2, label=f"alpha=2", linestyle='-', marker='o')
        plt.plot(loss3, label=f"alpha=4", linestyle='-', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Comparison for Different Alphas')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('../results/loss_comparison_d2.png')
        plt.show()
        print("Training completed successfully.")
        print("Loss comparison plot saved as 'loss_comparison_d2.png'.")
        