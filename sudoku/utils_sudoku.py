import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import time
import os
import argparse

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, Subset

import cvxpy as cp

def setup_cvx_qp_problem(opt_var_dim, num_ineq, num_eq):
    '''
    set up quadratic program functions for cvxpy
    '''
    Q_cp = cp.Parameter((opt_var_dim, opt_var_dim), PSD=True)
    p_cp = cp.Parameter(opt_var_dim)
    G_cp = cp.Parameter((num_ineq, opt_var_dim))
    h_cp = cp.Parameter(num_ineq)
    A_cp = cp.Parameter((num_eq, opt_var_dim))
    b_cp = cp.Parameter(num_eq)
    
    y_cp = cp.Variable(opt_var_dim)
    
    objective_fn = 0.5 * cp.sum_squares(Q_cp @ y_cp) + p_cp.T @ y_cp
    ineq_functions = [G_cp @ y_cp - h_cp]
    eq_functions = [A_cp @ y_cp -b_cp]
    ineq_constraints = [f<=0 for f in ineq_functions]
    eq_constraints = [f==0 for f in eq_functions]
    
    problem = cp.Problem(cp.Minimize(objective_fn), eq_constraints+ineq_constraints)
    assert problem.is_dpp()
    
    params = [Q_cp, p_cp, G_cp, h_cp, A_cp, b_cp]
    variables = [y_cp]
    
    return problem, objective_fn, ineq_functions, eq_functions, params, variables


def decode_onehot(encoded_board):
    """
    Take the unique argmax of the one-hot encoded board.
    
    encoded_board: (n**2, n**2, n**2), one-hot
        - dimensions correspond to (which_number, row, column)
    """
    v,I = torch.max(encoded_board, 0) #(row, column)
    return ((v>0).long()*(I+1)).squeeze()

def get_sudoku_matrix(n):
    '''
    sudoku board size is n**2 x n**2
    
    return the equality constraint matrix A for the LP of sudoku, the shape is (K, n**6) for some K
    '''
    X = np.array([[cp.Variable(n**2) for i in range(n**2)] for j in range(n**2)]) #(row, col, digit), row and column are square number
    cons = ([x >= 0 for row in X for x in row] + # one-hot (#row * #col * #digits constraints)
            [cp.sum(x) == 1 for row in X for x in row] + # each cell contains exactly one digit (#row * #col)
            [sum(row) == np.ones(n**2) for row in X] + # each row contains n**2 unique digits (#row * #digit)
            [sum([row[i] for row in X]) == np.ones(n**2) for i in range(n**2)] + # each column contains n**2 unique digit (#col * #digit)
            [sum([sum(row[i:i+n]) for row in X[j:j+n]]) == np.ones(n**2) for i in range(0,n**2,n) for j in range(0, n**2, n)]) # n**2 unique digits in n x n blocks (n**2 * n**2 * n**2)
    f = sum([cp.sum(x) for row in X for x in row])
    prob = cp.Problem(cp.Minimize(f), cons)

    A = np.asarray(prob.get_problem_data(cp.GUROBI)[0]["A"].todense())
    # print(f"A shape: {A.shape}")
    A0 = [A[0]]
    rank = 1
    for i in range(1,A.shape[0]):
        if np.linalg.matrix_rank(A0+[A[i]], tol=1e-12) > rank:
            A0.append(A[i])
            rank += 1

    return np.array(A0)

if __name__=="__main__":
    ######### board encoding #######
    # encoded_board = torch.zeros((9,4,4))
    # encoded_board[-1,:,:]=1
    # encoded_board[-1,1,1]=0
    # print(decode_onehot(encoded_board))
    
    ######### sudoku matrix #########
    A_new = get_sudoku_matrix(3)
    print(A_new.shape)