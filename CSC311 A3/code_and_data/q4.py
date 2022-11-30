'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    # for get_digits_by_label:
    # returns an y x 64 array, each row i represents the pixels of image i if that image matches query label

    for i in range(0, 10):
        matrix_label_i = data.get_digits_by_label(train_data, train_labels, i)
        ith_row = np.mean(matrix_label_i, axis=0)
        # Set axis=0 to add vertically, so we find mean for each column in the matrix (i.e. each pixel)

        # update the means matrix by changing its ith row to the above:
        means[i] = ith_row

    assert (means.shape == (10, 64))
    return means




def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances

    # Cov(x) = E[(x - mean)(x - mean)^T]
    # where x is the data vector for a particular training image

    N = train_labels.shape[0]
    means = compute_mean_mles(train_data, train_labels)

    for i in range(10):

        # Find the subset of images with label i:
        subset = data.get_digits_by_label(train_data, train_labels, i)
        y = subset.shape[0]
        sum_i = np.zeros((64, 64))

        # for every image j having label i:
        for j in range(y):
            A = (subset[j] - means[i]).reshape((64, 1))
            sum_i += A @ np.transpose(A)

        expectation = sum_i / y
        # (64, 64)
        covariances[i] = expectation + 0.01 * np.eye(64)

    return covariances






def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''

    n = digits.shape[0]
    d = 64  # num of pixels (features)

    # For ease of looping (loop from 0 to 9), first find the transpose:
    result_transpose = np.zeros((10, n))

    for k in range(10):
        mu_k = means[k].reshape((64, 1))
        sigma_k = covariances[k]    # Size 64 x 64
        det_sigma_k = np.linalg.det(sigma_k)
        inverse_sigma_k = np.linalg.inv(sigma_k)

        for i in range(n):
            x_i = digits[i].reshape((64, 1))
            entry_ki = (-d/2) * np.log(2 * np.pi) - (1/2) * np.log(det_sigma_k) \
                       - (1/2) * (np.transpose(x_i - mu_k) @ inverse_sigma_k @ (x_i - mu_k))

            result_transpose[k, i] = entry_ki

    # transpose back to n x 10
    result = np.transpose(result_transpose)
    return result





def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''

    # p(y|x, mu, sigma) = p(x, y|mu, sigma) / p(x|mu, sigma)
    #                   = p(x|y, mu, sigma) * p(y) / p(x|mu, sigma)
    #                   = p(x|y, mu, sigma) * (1/10) / p(x|mu, sigma)

    # where p(x|mu, sigma) = sum from i=0 to 9: p(x, y=i|mu, sigma)
    #                      = sum from i=0 to 9: p(x|y=i, mu, sigma) * p(y)

    gl = generative_likelihood(digits, means, covariances)
    n = digits.shape[0]
    log_y = np.log(1/10)
    result = np.zeros((n, 10))

    for i in range(n):
        # Find gl for the ith image:
        gl_i = gl[i].flatten()

        # Find p(x|mu, sigma):
        marginalized_i = gl_i[0]
        for j in range(1, 10):
            marginalized_i = np.logaddexp(marginalized_i, gl_i[j])
        marginalized_i += log_y

        for j in range(10):
            result[i, j] = gl_i[j] + log_y - marginalized_i

    return result






def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    n = digits.shape[0]
    correct_likelihood = 0
    for i in range(n):
        correct_label = labels[i]
        correct_likelihood += cond_likelihood[i][int(correct_label)]

    # Compute as described above and return
    return correct_likelihood / n





def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class

    # Return a list of length n
    n = digits.shape[0]
    result = np.zeros((n,))

    # for each image i, find its likelihoods:
    for i in range(n):
        ith_likelihoods = cond_likelihood[i]
        highest_index = np.argmax(ith_likelihoods)
        result[i] = highest_index

    return result







def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # print(data.get_digits_by_label(train_data, train_labels, 1).shape)
    # print(train_data.shape)
    # print(train_labels.shape)
    # assert(0)

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    avg_cond_train = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    avg_cond_test = avg_conditional_likelihood(test_data, test_labels, means, covariances)

    print(f"train avg conditional log-likelihood: {avg_cond_train}")
    print(f"test avg conditional log-likelihood: {avg_cond_test}")

    train_predictions = classify_data(train_data, means, covariances)
    train_accuracy = np.mean(train_predictions == train_labels)
    test_predictions = classify_data(test_data, means, covariances)
    test_accuracy = np.mean(test_predictions == test_labels)

    print(f"train accuracy: {train_accuracy}")
    print(f"test accuracy: {test_accuracy}")


    ############### Diagonal Covariances ################
    matrix_lst = []
    for i in range(10):
        cov_i = covariances[i]
        diag_cov_i = np.diag(np.diag(cov_i))
        matrix_lst.append(diag_cov_i)
    diag_covariances = np.array(matrix_lst)
    # print(diag_covariances.shape)

    avg_cond_train_diag = avg_conditional_likelihood(train_data, train_labels, means, diag_covariances)
    avg_cond_test_diag = avg_conditional_likelihood(test_data, test_labels, means, diag_covariances)

    print(f"train avg conditional log-likelihood (diagonal covariances): {avg_cond_train_diag}")
    print(f"test avg conditional log-likelihood (diagonal covariances): {avg_cond_test_diag}")

    train_predictions_diag = classify_data(train_data, means, diag_covariances)
    train_accuracy_diag = np.mean(train_predictions_diag == train_labels)
    test_predictions_diag = classify_data(test_data, means, diag_covariances)
    test_accuracy_diag = np.mean(test_predictions_diag == test_labels)

    print(f"train accuracy (diag): {train_accuracy_diag}")
    print(f"test accuracy (diag): {test_accuracy_diag}")






if __name__ == '__main__':
    main()
