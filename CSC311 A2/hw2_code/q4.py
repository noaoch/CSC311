# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
np.random.seed(0)

import scipy

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:] 
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO

    # Notation: A[i,:] means the ith row of matrix A
    
    # Convert test_datum into a 1xd matrix
    d = test_datum.shape[0]
    datum_t = test_datum.reshape(1, d)

    # Output a Nx1 matrix where value at index i is the square norm between ith row of x_train and test_datum
    num_top = l2(datum_t, x_train)

    num_bot = 2 * (tau ** 2)
    num = np.exp(-1 * num_top / num_bot)

    # Find sum of every element in the Nx1 matrix above (num)
    denom = 0
    for j in range(len(x_train)):
       denom += num[0][j]

    weights = num / denom  # weights is also a Nx1 matrix where value at index i is a^(i)

    # Compute optimal weights
    # Solve the eqn: (X^TAX + lam * I)w = X^TAy

    left = np.transpose(x_train) @ np.diag(weights.flatten()) @ x_train + lam * np.eye(d, d)
    right = np.transpose(x_train) @ np.diag(weights.flatten()) @ y_train

    w_optimal = np.linalg.solve(left, right)
    y_hat = np.dot(test_datum, w_optimal)


    return y_hat

    ## TODO




def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    ## TODO
    
    train_inputs, valid_inputs, train_label, valid_label = train_test_split(x, y, test_size=val_frac)

#     # Vectorize the targets so that it is suitable for LRLS (shape is Nx1)
#     train_targets = train_targets.reshape(-1, 1)
#     valid_targets = valid_targets.reshape(-1, 1)

#     train_loss_result = np.zeros((len(taus), 1))
#     valid_loss_result = np.zeros((len(taus), 1))
    
    train_result = []
    valid_result = []

    for j in range(len(taus)): # for every taus:
        total_train_loss = 0
        # loop over the entire training set
        for i in range(len(train_inputs)):
            y_hat = LRLS(train_inputs[i], train_inputs, train_label, taus[j])
            t = train_label[i]
            total_train_loss += (y_hat - t) ** 2
       
        train_loss = total_train_loss / (2 * len(train_inputs))  # Loss is ||Xw - t||^2 / 2N
        train_result.append(train_loss)


        total_valid_loss = 0
        # loop over the validation set
        for i in range(len(valid_inputs)):
            y_hat = LRLS(valid_inputs[i], train_inputs, train_label, taus[j])
            t = valid_label[i]
            total_valid_loss += (y_hat - t) ** 2

        valid_loss = total_valid_loss / (2 * len(valid_inputs))
        valid_result.append(valid_loss)

    train_result = np.array(train_result).reshape(-1, 1)
    valid_result = np.array(valid_result).reshape(-1, 1)
    return train_result, valid_result



    ## TODO


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    print("Begin running")
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    print("Begin plotting")
    plt.semilogx(train_losses)
    plt.title("training loss")
    plt.show()

    plt.semilogx(test_losses)
    plt.title("validation loss")
    plt.show()


