from gettext import bind_textdomain_codeset
import secrets
from utils import sigmoid

import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO:                                                             #
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################

    # z = Xw + b, where w is the vector of weights, X is the data matrix, b is the bias/intercept
    # Then y = sigmoid(z), where y is the prediction 'squashed' into range [0,1]

    bias = weights[-1]

    weights_no_bias = weights[:-1] # disregard the last element as it is the bias

    z = np.matmul(data, weights_no_bias) + bias  # note bias is added to every element in the array
    y = sigmoid(z)

    


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################

    total_ce = 0
    accurate_predictions = 0

    for i in range(len(targets)):
        # Implementing logistic regression naively can cause numerical instabilities
        # So use different algorithm involving z, not y

        # Inverse sigmoid function: z = ln(y/(1-y))
        y_i = y[i][0]
        z_i = np.log(y_i/(1-y_i))
        t_i = targets[i][0]

        # our algorithm (from lec05)
        total_ce += t_i * np.logaddexp(0, -z_i) + (1 - t_i) * np.logaddexp(0, z_i)

        if y_i >= 0.5: 
            predicted = 1
        else:
            predicted = 0
        
        # compare predicted with target
        if predicted == t_i:
            accurate_predictions += 1
    
    ce = total_ce / len(targets)
    frac_correct = accurate_predictions / len(targets)


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################

    f = evaluate(targets, y)[0]

    df_lst = []

    # Evaluate the gradient w.r.t each w_j
    for j in range(len(weights) - 1): # traverse matrix horizontally (M traverses)

        sum_i = 0  # total sum of gradients over 1 column
        for i in range(len(targets)): # traverse vertically (N traverses)
            x_ij = data[i, j]
            sum_i += (y[i][0] - targets[i][0]) * x_ij
        
        # compute the average of the sum over entire column i
        avg_sum_i = sum_i / len(targets)
        df_lst.append([avg_sum_i])


    # Finally, evaluate gradient w.r.t. bias
    # dL / db = y - t

    bias_grad = 0
    for i in range(len(targets)):
        bias_grad += y[i][0] - targets[i][0]

    avg_bias_grad = bias_grad / len(targets)
    df_lst.append([avg_bias_grad])

    df = np.array(df_lst)


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y


