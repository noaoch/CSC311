from l2_distance import l2_distance
from utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################
    
    x = [1,3,5,7,9]
    y = []

    for k in x:
        predicted = knn(k, train_inputs, train_targets, valid_inputs)

        assert(len(predicted) == len(valid_targets))

        correctly_predicted = 0
        for i in range(len(predicted)):
            if predicted[i] == valid_targets[i]:
                correctly_predicted += 1
        
        accuracy = correctly_predicted / len(valid_targets)
        y.append(accuracy)

    plt.plot(x,y,label='validation')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()

    # 3(b):
    # Choose k* = 5 as it provides the highest accuracy of 0.860 on the validation set, and we can
    # conveniently access the validation accuracies of k*, k* - 2 and k* + 2.
    x_star = [3,5,7]
    y_valid = y[1:4]
    y_test = []

    for k in x_star:
        test_result = knn(k, train_inputs, train_targets, test_inputs)

        assert(len(test_result) == len(test_targets))

        correctly_predicted_test = 0
        for i in range(len(test_targets)):
            if test_result[i] == test_targets[i]:
                correctly_predicted_test += 1
        
        test_accuracy = correctly_predicted_test / len(test_targets)
        y_test.append(test_accuracy)

    plt.plot(x_star, y_valid, label='validation')
    plt.plot(x_star, y_test, label='test')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
    







    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_knn()
