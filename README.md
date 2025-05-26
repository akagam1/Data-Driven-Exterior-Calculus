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


