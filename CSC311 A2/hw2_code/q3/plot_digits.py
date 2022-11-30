import numpy as np
import matplotlib.pyplot as plt


def plot_digits(digit_array):
    """ Visualizes each example in digit_array.
    :param digit_array: N x M array of pixel intensities.
    :return: None
    """
    CLASS_EXAMPLES_PER_PANE = 5

    # assume two evenly split classes
    examples_per_class = int(digit_array.shape[0] / 2)
    num_panes = int(np.ceil(float(examples_per_class) / CLASS_EXAMPLES_PER_PANE))

    for pane in range(num_panes):
        print("Displaying pane {}/{}".format(pane + 1, num_panes))

        top_start = pane * CLASS_EXAMPLES_PER_PANE
        top_end = min((pane + 1) * CLASS_EXAMPLES_PER_PANE, examples_per_class)
        top_pane_digits = extract_digits(digit_array, top_start, top_end)

        bottom_start = top_start + examples_per_class
        bottom_end = top_end + examples_per_class
        bottom_pane_digits = extract_digits(digit_array, bottom_start, bottom_end)

        show_pane(top_pane_digits, bottom_pane_digits)


def extract_digits(digit_array, start_index, end_index):
    """ Returns a list of 28 x 28 pixel intensity arrays starting
    at start_index and ending at end_index.
    """

    digits = []
    for index in range(start_index, end_index):
        digits.append(extract_digit_pixels(digit_array, index))

    return digits


def extract_digit_pixels(digit_array, index):
    """ Extracts the 28 x 28 pixel intensity array at the specified index.
    """
    return digit_array[index].reshape(28, 28)


def show_pane(top_digits, bottom_digits):
    """ Displays two rows of digits on the screen.
    """
    all_digits = top_digits + bottom_digits
    fig, axes = plt.subplots(nrows=2, ncols=int(len(all_digits) / 2))
    for axis, digit in zip(axes.reshape(-1), all_digits):
        axis.imshow(digit, interpolation='nearest', cmap=plt.gray())
        axis.axis('off')
    plt.tight_layout()
    plt.show()
