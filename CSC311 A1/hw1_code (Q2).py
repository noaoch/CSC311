from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import numpy as np



###############   2a   ###################

def load_data(realPath, fakePath):
    vectorizer = CountVectorizer()

    real_file = open(realPath, "r")
    realData = real_file.readlines()  # Output a list of all the lines in realFile
    realLabels = [1] * len(realData)  # Create the y labels for realData (1 denotes real)

    fakeFile = open(fakePath, "r")
    fakeData = fakeFile.readlines()  # List all lines in fakeFile
    fakeLabels = [0] * len(fakeData)  # Create the y labels for fakeData (0 denotes fake)

    allData = realData + fakeData  # Join datas together
    y = realLabels + fakeLabels  # Join labels together


    # Turn data into an array
    X = vectorizer.fit_transform(allData)
    feature_name = vectorizer.get_feature_names_out()

    
    # Split to 70% training and 30% testing/validation
    train_data, test_valid_data, train_label, test_valid_label = train_test_split(X, y, test_size=0.3, train_size=0.7)

    # Split the test_valid_data to 50% testing and 50% validation, so 70% training, 15% testing and 15% validation
    test_data, valid_data, test_label, valid_label = train_test_split(test_valid_data, test_valid_label, test_size=0.5)


    return train_data, test_data, valid_data, train_label, test_label, valid_label, feature_name





##################   2b   ###################

def select_model(train_data, train_label, valid_data, valid_label):

    # Setup
    max_depths = [5, 10, 15, 20, 25]
    giniScore_lst = []
    entropyScore_lst = []
    logScore_lst = []


    for d in max_depths: # max_depth contains at least 5 different depths
        
        # Initialize a tree based on gini, entropy and log loss
        giniTree = DecisionTreeClassifier(criterion="gini", max_depth=d) 
        entropyTree = DecisionTreeClassifier(criterion="entropy", max_depth=d) 
        logTree = DecisionTreeClassifier(criterion="log_loss", max_depth=d)

        # Build the decision trees with the training set
        fitted_giniTree = giniTree.fit(train_data, train_label) 
        fitted_entropyTree = entropyTree.fit(train_data, train_label)
        fitted_logTree = logTree.fit(train_data, train_label)


        # Evualuate the accuracies of each model on the validation set
        giniScore = fitted_giniTree.score(valid_data, valid_label)
        giniScore_lst.append(giniScore)
        print(f"Gini: score = {giniScore}, depth = {d}")

        entropyScore = fitted_entropyTree.score(valid_data, valid_label)
        entropyScore_lst.append(entropyScore)
        print(f"Information gain: score = {entropyScore}, depth = {d}")

        logScore = fitted_logTree.score(valid_data, valid_label)
        logScore_lst.append(logScore)
        print(f"Log loss: score = {logScore}, depth = {d}")


    # Plotting
    plt.plot(max_depths, giniScore_lst, 'r', label='Gini')
    plt.plot(max_depths, entropyScore_lst, 'g', label='Info Gain')
    plt.plot(max_depths, logScore_lst, 'c', label='Log loss')

    plt.legend()
    plt.show()



#################   2c   ###################

def draw_tree(train_data, train_label, feature_names):
    # The best model is a giniTree with depth = 25
    giniTree = DecisionTreeClassifier(criterion="gini", max_depth=25)
    fitted_giniTree = giniTree.fit(train_data, train_label)

    export_graphviz(fitted_giniTree, out_file="best_tree.dot", max_depth=2, \
        feature_names= feature_names, class_names=["fake", "real"]) # fake class first, real class second





#################   2d   ####################

# IG(Y|X) = H(Y) - H(Y|X)

def entropy_before_split(train_label):
    # calculate H(Y), where Y in {real, fake}

    real_count = 0
    fake_count  = 0

    for i in train_label:
        if i == 1:
            real_count += 1
        else:
            fake_count += 1

    # Compute P(Y = fake) and P(Y = real)
    fakeProb = fake_count / len(train_label)
    realProb = real_count / len(train_label)

    # H(Y) = - summation over Y: P(y)log_2(P(y))
    entropy_Y = -(math.log(fakeProb, 2) * fakeProb) - (math.log(realProb, 2) * realProb)

    return entropy_Y





def entropy_after_split(train_data, train_label, keyword):
    # calculate H(Y|X), where Y in {real, fake}, and X in {keyword present, not present}
    # H(Y|X) = - summation over X: summation over Y: P(x,y)log_2(P(y|x))

    # I will use X to denote keyword being present, and no_X to denote keyword not present

    real_X_count = 0  # count of news where keyword is present, and is real
    fake_X_count = 0

    real_no_X_count = 0
    fake_no_X_count = 0

    keyword_index = feature_name.tolist().index(keyword)  
    # get the index of the keyword in the list of feature names

    # Note: train_data is a sparse matrix that has the following structure:
    # row i = the ith headline
    # column j = the jth keyword
    # train_data[i, j] = the number of times jth keyword appeared in the ith headline

    for index in range(len(train_label)):
        if train_data[index, keyword_index] != 0: # if keyword is present in the sentence
            if train_label[index] == 1: # if Y = real:
                real_X_count += 1
            else:                       # if Y = fake:
                fake_X_count += 1
        
        else: # else keyword is not present
            if train_label[index] == 1:
                real_no_X_count += 1
            else:
                fake_no_X_count += 1

    # Compute P(Y, X) for all cases:
    # fake and present, real and resent, fake and not present, real and not present
    prob_fake_and_X = fake_X_count / len(train_label)
    prob_real_and_X = real_X_count / len(train_label)

    prob_fake_no_X = fake_no_X_count / len(train_label)
    prob_real_no_X = real_no_X_count / len(train_label)


    # Compute P(keyword) and P(no keyword)
    total_X_count = real_X_count + fake_X_count
    prob_X = total_X_count / len(train_label)

    total_no_X_count = real_no_X_count + fake_no_X_count
    prob_no_X = total_no_X_count / len(train_label)


    # Compute P(Y|X)
    prob_fake_given_X = prob_fake_and_X / prob_X
    prob_real_given_X = prob_real_and_X / prob_X
    
    prob_fake_given_no_X = prob_fake_no_X / prob_no_X
    prob_real_given_no_X = prob_real_no_X / prob_no_X

    # compute the expected conditional entropy H(Y|X)
    entropy_Y_given_X = -(math.log(prob_fake_given_X, 2) * prob_fake_and_X) \
        - (math.log(prob_real_given_X, 2) * prob_real_and_X) \
            - (math.log(prob_fake_given_no_X, 2) * prob_fake_no_X) \
                - (math.log(prob_real_given_no_X, 2) * prob_real_no_X)

    return entropy_Y_given_X
        
    



def compute_information_gain(train_data, train_label, keyword):

    # IG(Y|X) = H(Y) - H(Y|X)
    entropy_Y = entropy_before_split(train_label) 
    entropy_Y_given_X = entropy_after_split(train_data, train_label, keyword)
    print(f"IG(Y|X) is {entropy_Y - entropy_Y_given_X} for the keyword {keyword}") 





if __name__ == "__main__":
    train_data, test_data, valid_data, train_label, test_label, valid_label, feature_name \
        = load_data("clean_real.txt", "clean_fake.txt")

    # Find the best model and depth
    select_model(train_data, train_label, valid_data, valid_label)

    # Export the tree to a file "best_tree.dot", which can be converted to png by 
    # "dot -Tpng best_tree.dot -o best_tree.png"
    draw_tree(train_data, train_label, feature_name)

    # Compute information gain
    compute_information_gain(train_data, train_label, "the")

    compute_information_gain(train_data, train_label, "hillary")

    compute_information_gain(train_data, train_label, "trumps")

    compute_information_gain(train_data, train_label, "donald")