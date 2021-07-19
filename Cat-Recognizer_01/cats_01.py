#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Program determines if the given image represents a cat. This determination
is executed by a logistic regression machine learning model.
"""

__author__ = "Jesse Deda"

import numpy as np
import h5py as h5py
import copy
import sys
from PIL import Image


def preprocess_dataset(train_set_x_orig, test_set_x_orig, train_set_y_orig, test_set_y_orig):
    """
    Flattens and standardizes the training sets X and reshapes Y.

    :param train_set_x_orig: set of training input feature vectors (images)
    :param test_set_x_orig: set of testing input feature vectors (images)
    :param train_set_y_orig: set of training predicted labels (0 or 1)
    :param test_set_y_orig: set of testing predicted labels (0 or 1)

    :return: flattened and standardized data set
    """

    # Flatten sets X by converting 3D shaped array into a  column vector.
    # The numpyarray (numExamples, height, width, colormdl) converts to (height * width * colormdl, numExamples)
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    # Center sets X by dividing each pixel intensity by the maximum pixel intensity valuee 255.
    train_set_x_final = train_set_x_flatten / 255.0
    test_set_x_final = test_set_x_flatten / 255.0

    # Reshape sets Y into row vectors.
    train_set_y_final = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_final = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    # Return preprocessed dataset.
    return train_set_x_final, test_set_x_final, train_set_y_final, test_set_y_final


def load_dataset():
    """
    Loads and returns the model's training and testing set and classes from the h5 files.

    :return: training and testing set x and y, and classes
    """

    # Initialize training sets (h5 to np.array).
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    # Initialize testing sets (h5 to np.array).
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    # Initialize classes.
    classes = np.array(test_dataset["list_classes"][:])

    # Record shapes of data sets.
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]  # height = weight
    num_clr = train_set_x_orig.shape[3]  # rgb, cmyk, etc ...

    # Preprocess data sets.
    train_set_x_final, test_set_x_final, train_set_y_final, test_set_y_final = preprocess_dataset(train_set_x_orig,
                                                                                                  test_set_x_orig,
                                                                                                  train_set_y_orig,
                                                                                                  test_set_y_orig)

    # Return all data sets and class.

    return m_train, m_test, num_px, num_clr, train_set_x_final, test_set_x_final, train_set_y_final, test_set_y_final, classes


def init_zeros(dim):
    """
    Initializes weights (w) and bias (b) with zeros.

    :param dim: the dimension of the column vector w.

    :return: w and b.
    """

    w = np.zeros((dim, 1))
    b = 0.0
    return w, b


def sigmoid(z):
    """
    Computes the sigmoid of the given value.

    :param z: the given value

    :return: the sigmoided value
    """

    return 1 / (1 + np.exp(-z))


def propagate(w, b, X, Y):
    """
    Computes and returns the cost of the cost function and its
    gradients with respect to w and b.

    :param w: weights
    :param b: bias
    :param X: training set of input feature vectors
    :param Y: training set of labels

    :return: cost and gradients dw, db
    """

    # Compute cost (forward propagation).
    m = X.shape[1]
    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    loss = -(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
    cost = (1.0 / m) * np.sum(loss)
    cost = np.squeeze(np.array(cost))

    # Compute gradients (backward propagation).
    dz = A - Y
    dw = (1.0 / m) * np.dot(X, dz.T)
    db = (1.0 / m) * np.sum(dz)
    gradients = {"dw": dw, "db": db}

    # Return gradients and costs to be optimized.
    return gradients, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    """
    Optimizes parameters w, b via gradient descent.

    :param w: weights
    :param b: bias
    :param X: training set of input feature vectors
    :param Y: training set of labels
    :param num_iterations: number of gradient descent steps
    :param learning_rate: descent step size
    :param print_cost: true or false ot print cost of function over iterations

    :return: optimized parameters, the gradients, and cost
    """

    # Clone model parameters.
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    # Initialize dw and db.
    dw = 0.0
    db = 0.0

    # Accumulate costs.
    costs = []

    for i in range(num_iterations):

        # Propagate.
        gradients, cost = propagate(w, b, X, Y)

        # Update model parameters.
        dw = gradients["dw"]
        db = gradients["db"]
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)

        # Record costs every 100 iterations.
        if i % 100 == 0:
            costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    # Return parameters, gradients, and costs.
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs


def predict(w, b, X):
    """
    Predicts the label for each vector of the set X.

    :param w: weights
    :param b: bias
    :param X: training set of input feature vectors

    :return: labeled set Y
    """

    # Initialize prediction set as a zero row vector and weights as zero column vector.
    m = X.shape[1]
    Y = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Calculate activation matrix (probabilities image is a cat).
    A = sigmoid(np.dot(w.T, X) + b)

    # Decision boundary at 0.5. Initialize prediction set.
    for i in range(A.shape[1]):

        if A[0, i] > 0.5:
            Y[0, i] = 1
        else:
            Y[0, i] = 0

    # Return predicted set.
    return Y


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Trains and tests model accuracy.

    :param X_train: training set of input feature vectors
    :param Y_train: training set of labels
    :param X_test: test set of input feature vectors
    :param Y_test: test set of labels
    :param num_iterations: number of gradient descent steps
    :param learning_rate: descent step size
    :param print_cost: true or false ot print cost of function over iterations

    :return: costs, label sets, w, b, rate, and its.
    """

    if print_cost:
        print("\nTraining ...")

    # Find parameters w and b (train model).
    w, b = init_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]

    # Collect model predictions.
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print training and test accuracy.
    if print_cost:
        print("\nTrain accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("Test accuracy: {} %\n".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    # Return costs, label sets, w, b, rate, and its.
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d


def main(argv):
    """
    Runs the program. User may enter an image and the will prompted by the cat recognizer whether or not
    the given image is a cat picture or not.

    :param argv: image name

    :return: 0, exit success otherwise failure
    """

    # Initialize model parameters.
    m_train, m_test, num_px, num_clr, x_train_set, x_test_set, y_train_set, y_test_set, classes = load_dataset()
    iterations = 2000
    learning_rate = 0.005
    print_cost = True

    # Train model.
    logistic_regression_model = model(x_train_set, y_train_set, x_test_set, y_test_set, iterations, learning_rate,
                                      print_cost)

    # Preprocess and predict.
    print("Filename: " + argv)
    img_name = "user_images/" + argv
    image = np.array(Image.open(img_name).resize((num_px, num_px)))
    image = image / 255.0
    image = image.reshape((1, num_px * num_px * num_clr)).T
    my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)
    print("This is a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.\n")

    return 0


if __name__ == '__main__':
    """
    Calls main (runs program).
    """

    main(sys.argv[1])
