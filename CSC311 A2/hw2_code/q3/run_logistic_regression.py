from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.3,
        "weight_regularization": 0.,
        "num_iterations": 200
    }
    weights = np.zeros((M + 1, 1))  # init all weights to 0
    initial_weight = weights[0][0]

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################

    test_inputs, test_targets = load_test()
    train_loss = []
    valid_loss = []

    for t in range(hyperparameters["num_iterations"]):
        
        train_avg_loss, train_df, train_y = logistic(weights, train_inputs, train_targets, hyperparameters)

        valid_avg_loss, valid_df, valid_y = logistic(weights, valid_inputs, valid_targets, hyperparameters)

        train_loss.append(train_avg_loss)
        valid_loss.append(valid_avg_loss)


        # Update weights via gradient descent
        weights = weights - hyperparameters["learning_rate"] * train_df

        if t == hyperparameters["num_iterations"] - 1:
            test_avg_loss, test_df, test_y = logistic(weights, test_inputs, test_targets, hyperparameters)

            final_ce_train, final_accuracy_train = evaluate(train_targets, train_y)
            final_ce_valid, final_accuracy_valid = evaluate(valid_targets, valid_y)
            final_ce_test, final_accuracy_test = evaluate(test_targets, test_y)

            print(
            f"""
            Train: cross entropy = {final_ce_train}, accuracy = {final_accuracy_train}, 
            Validation: cross entropy = {final_ce_valid}, accuracy = {final_accuracy_valid}, 
            Test: cross entropy = {final_ce_test}, accuracy = {final_accuracy_test}
            """)

    
    x = list(range(1, hyperparameters["num_iterations"] + 1))
    plt.plot(x, train_loss, label="training loss")
    plt.plot(x, valid_loss, label="validation loss")
    plt.legend()
    plt.title(
        f"""Result with learning rate = {hyperparameters['learning_rate']}, 
        initial weight = {initial_weight}, 
        # iterations = {hyperparameters['num_iterations']}""")
    plt.show()

    






    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
