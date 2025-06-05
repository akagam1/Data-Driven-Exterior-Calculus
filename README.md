# Data-Driven-Exterior-Calculus

This is an implementation of PDE Constrained Optimization, using the Data Driven Exterior Calculus framework presented in this [paper](https://www.sciencedirect.com/science/article/pii/S0021999122000316?ref=pdf_download&fr=RR-2&rr=91c8010afddfff9b) (Trask et al., 2022). 

## Setup

Made use of a conda environment to manage the packages and libraries required. Setup a conda environment, activate it, and install requirements as follows:
```
conda create -n ddec_env python=3.8
conda activate ddec_env
pip3 install -r requirements.txt
```

## Training and Testing for Darcy Flow

```
python run.py --N={Grid Size} --alpha={boundary condition} --iter={Iterations} --tol={Iterative Solver Tolerance} --epsilon={Non-Linear weight} --epochs={Training Epochs} --problem_type={Darcy D1/D2} --lr={Learning Rate}
```

## Darcy problem D1 results

The following are the results for the D1 problem of the paper over different domain sizes. Clearly the linear model trains very well, with the loss falling rapidly.

<img title="6x6 Domain" src="results/N=6_D1_results.png">

<img title="8x8 Domain" src="results/N=8_D1_results.png">

<img title="10x10 Domain" src="results/N=10_D1_results.png">

<img title="Domain model loss comparison" src="results/d1loss_comparison.png">

## Darcy problem D2 results


<!-- 
The following are the results for the D2 problem of the paper. The model was trained with a batch size of 3 for different boundary conditions ($\alpha \in \{1,2,4\}$).

<img  src="results/N=6_alpha=1_D2_results.png">

<img  src="results/N=6_alpha=2_D2_results.png">

<img  src="results/N=6_alpha=4_D2_results.png"> -->

