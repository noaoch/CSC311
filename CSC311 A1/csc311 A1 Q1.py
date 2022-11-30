from array import array
import numpy as np
import matplotlib.pyplot as plt

def main():

    #######  Setup for plotting  #######
    l2_avg_pts = []
    l2_std_pts = []

    l1_avg_pts = []
    l1_std_pts = []

    dim_pts = []
    ###############################

    for i in range(0, 11):

        # set up the empty list of distances for the current dimension
        l2_dist = []
        l1_dist = []

        dim = 2 ** i
        dim_pts.append(dim)

        # produce a list of 100 vectors each with dimension dim
        vector_lst = np.random.rand(100, dim)


        print("computing distances")
        # compute the distances & populate l2_dist and l1_dist
        distance(vector_lst.tolist(), l2_dist, l1_dist)
        
        # Both l2_dist and l1_dist have len 4950, which is 100 choose 2

        l2_avg = np.mean(l2_dist)
        l2_avg_pts.append(l2_avg)
        l2_std = np.std(l2_dist)
        l2_std_pts.append(l2_std)

        l1_avg = np.mean(l1_dist)
        l1_avg_pts.append(l1_avg)
        l1_std = np.std(l1_dist)
        l1_std_pts.append(l1_std)


    #############  Plotting  #############

    figure, axis = plt.subplots(2,2) # initialize the subplot to have width 2 and height 2

    axis[0,0].plot(dim_pts, l2_avg_pts)
    axis[0,0].set_title("l2 distance mean")

    axis[0,1].plot(dim_pts, l2_std_pts)
    axis[0,1].set_title("l2 distance SD")

    axis[1,0].plot(dim_pts, l1_avg_pts)
    axis[1,0].set_title("l1 distance mean")

    axis[1,1].plot(dim_pts, l1_std_pts)
    axis[1,1].set_title("l1 distance SD")

    # display
    plt.show()
    ######################################
        


def distance(vec_lst, l2_dist, l1_dist):
    """
    computes the l1/l2 distance of all pairs of vectors in vec_lst, which is a list of
    vectors, then populate the list l2_dist with all l2 distances, and l1_dist with all l1 
    distances.

    Precondition: - all vectors in vec_lst must have the same dimension
                  - l2_dist and l1_dist are empty arrays
    """

    for i in range(len(vec_lst)):
        # start with a vector
        vec1 = vec_lst[i]

        # pair vec1 with every following vector
        for j in range(i + 1, len(vec_lst)):
            vec2 = vec_lst[j]

            # compute l2 and l1 distances between vec1 and vec2
            l2_sum = 0
            l1_sum = 0
            for entry_index in range(len(vec1)):
                l2_sum += (vec1[entry_index] - vec2[entry_index]) ** 2
                l1_sum += abs(vec1[entry_index] - vec2[entry_index])

            l2_dist.append(l2_sum)
            l1_dist.append(l1_sum)


if __name__ == "__main__":
    print("Hello")
    main()


