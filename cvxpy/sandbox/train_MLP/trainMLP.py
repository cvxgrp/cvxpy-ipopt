import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

class TrainMLP():
    """
    Train Multi-layer-perceptron.
        Params:
            [W_1, b_1, W_2, b_2]: (p + 3)-vector

        Ex:
            p = 8
    """
    def __init__(self, F_mat, z_vec, r_vec):
        self.F_mat = F_mat  # T by p matrix
        self.z_vec = z_vec  # T by 1 vector
        self.r_vec = r_vec  # T by 1 vector
        self.T, self.p = F_mat.shape

    def loss_func(self):
        W_1 = cp.Variable((1, self.p))  # 1 by p vector
        b_1 = cp.Variable(1)     
        W_2 = cp.Variable(1)    
        b_2 = cp.Variable(1)     

        MLP_vec = W_2 * cp.maximum(0, self.F_mat @ W_1.T + b_1) + b_2  # T by 1 vector
        MLP_vec = cp.reshape(MLP_vec, (self.T,))
        objective = cp.Minimize(cp.sum(cp.multiply(self.z_vec, cp.square(self.r_vec - MLP_vec))))
        # No constraint in my problem. Consider it here for optimization purpose.
        constraints = [MLP_vec >=0 , MLP_vec <= 1]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True)

        return W_1.value, b_1.value, W_2.value, b_2.value

    def predict(self, F_new, W_1, b_1, W_2, b_2):
        hidden = np.maximum(0, F_new @ W_1.T + b_1)
        predictions = W_2 * hidden + b_2
        return predictions


F_mat = pd.read_pickle("/Users/willizz/Documents/cvxpy-ipopt/cvxpy/sandbox/train_MLP/F_mat.pkl")
z_vec = pd.read_pickle("/Users/willizz/Documents/cvxpy-ipopt/cvxpy/sandbox/train_MLP/z_vec.pkl")
r_vec_1 = pd.read_pickle("/Users/willizz/Documents/cvxpy-ipopt/cvxpy/sandbox/train_MLP/r_vec_1.pkl")
r_vec_3 = pd.read_pickle("/Users/willizz/Documents/cvxpy-ipopt/cvxpy/sandbox/train_MLP/r_vec_3.pkl")
r_vec_5 = pd.read_pickle("/Users/willizz/Documents/cvxpy-ipopt/cvxpy/sandbox/train_MLP/r_vec_5.pkl")
r_vec_10 = pd.read_pickle("/Users/willizz/Documents/cvxpy-ipopt/cvxpy/sandbox/train_MLP/r_vec_10.pkl")
r_vec_50 = pd.read_pickle("/Users/willizz/Documents/cvxpy-ipopt/cvxpy/sandbox/train_MLP/r_vec_50.pkl")

F_mat = F_mat.to_numpy()
z_vec = z_vec.to_numpy().flatten()
r_vec_1 = r_vec_1.to_numpy().flatten()
r_vec_3 = r_vec_3.to_numpy().flatten()
r_vec_5 = r_vec_5.to_numpy().flatten()
r_vec_10 = r_vec_10.to_numpy().flatten()
r_vec_50 = r_vec_50.to_numpy().flatten()

MLP = TrainMLP(F_mat, z_vec, r_vec_1)
W_1, b_1, W_2, b_2 = MLP.loss_func()
pred = MLP.predict(F_mat, W_1, b_1, W_2, b_2)
#print(pred[:20])
print("W_1, b_1, W_2, b_2:")
print(W_1, b_1, W_2, b_2)
