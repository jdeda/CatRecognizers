#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Program determines if the given image represents a cat. This determination
is executed by a deep neural network model.
"""

__author__ = "Jesse Deda"

import numpy as np
import h5py as h5py
import copy
import sys
from PIL import Image


def dataset_preprocess(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig):
    """
    Flattens and standardizes the training sets X and reshapes Y.

    :param X_train_orig: set of training input feature vectors (images)
    :param Y_train_orig: set of testing input feature vectors (images)
    :param X_test_orig: set of training predicted labels (0 or 1)
    :param Y_test_orig: set of testing predicted labels (0 or 1)

    :return: flattened and standardized data set
    """

    # Flatten X sets.
    X_train_flat = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flat = X_test_orig.reshape(X_test_orig.shape[0], -1).T

    # Center sets X by dividing each pixel intensity by the maximum pixel intensity value 255.
    X_train_final = X_train_flat / 255.0
    X_test_final = X_test_flat / 255.0

    # Reshape sets Y into row vectors.
    Y_train_final = Y_train_orig.reshape((1, Y_train_orig.shape[0]))
    Y_test_final = Y_test_orig.reshape((1, Y_test_orig.shape[0]))

    # Return preprocessed dataset.
    return X_train_final, Y_train_final, X_test_final, Y_test_final


def dataset_load():
    """
    Loads and returns the model's training and testing set and classes from the h5 files.

    :return: training and testing set x and y, and classes
    """

    # Initialize training sets (h5 to np.array).
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    X_train_orig = np.array(train_dataset["train_set_x"][:])
    Y_train_orig = np.array(train_dataset["train_set_y"][:])

    # Initialize testing sets (h5 to np.array).
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    X_test_orig = np.array(test_dataset["test_set_x"][:])
    Y_test_orig = np.array(test_dataset["test_set_y"][:])

    # Initialize classes.
    classes = np.array(test_dataset["list_classes"][:])

    # Record shapes of data sets.
    m_train = X_train_orig.shape[0]
    m_test = X_test_orig.shape[0]
    num_px = X_train_orig.shape[1]  # height = weight
    num_clr = X_train_orig.shape[3]  # rgb, cmyk, etc ...

    # Preprocess data sets.
    X_train_final, Y_train_final, X_test_final, Y_test_final = dataset_preprocess(X_train_orig, Y_train_orig,
                                                                                  X_test_orig, Y_test_orig)

    # Return all data sets and class.
    dataset = {
        "m_train": m_train,
        "m_test": m_test,
        "num_px": num_px,
        "num_clr": num_clr,
        "X_train": X_train_final,
        "Y_train": Y_train_final,
        "X_test": X_test_final,
        "Y_test": Y_test_final,
        "classes": classes
    }
    return dataset


def nn_h_params_init():
    """
    Initializes hyper parameters for the model.
    :return: hyper parameters
    """

    h_params = {
        "n_iterations": 2500,
        "l_rate": 0.0075,
        "l_dims": [12288, 20, 7, 5, 1],
        "print_cost": True
    }
    return h_params


def nn_params_init(l_dims):
    """
    Initializes the parameters of the model. Each layer has its own
    set of parameters W and b.
    :param l_dims: dimensions of the layers of the model
    :return: parameters of the model
    """

    np.random.seed(1)
    params = {}
    L = len(l_dims)
    for l in range(1, L):
        params["W" + str(l)] = np.random.randn(l_dims[l], l_dims[l - 1]) / np.sqrt(l_dims[l - 1])  # *0.01
        params["b" + str(l)] = np.zeros((l_dims[l], 1))

    return params


def relu(Z):
    """
    Computes and returns A (the rectified linear output of Z) along with caches (Z).
    :param Z: value to compute relu of
    :return: A, cache
    """

    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = Z
    return A, cache


def sigmoid(Z):
    """
    Computes and returns A (the sigmoid of Z along) with caches (Z).
    :param Z: value to compute sigmoid of
    :return: A, cache
    """

    A = 1.0 / (1.0 + np.exp(-Z))
    cache = Z
    return A, cache


def l_forward_linear_activation(W, A_prev, b):
    """
    Computes and returns linear activation Z of the layer along with caches (W, A_prev, b).
    :param W: weights of layer
    :param A_prev: activations of previous layer
    :param b: biases of previous layer
    :return: Z, caches (W, A_prev, b)
    """

    Z = np.dot(W, A_prev) + b
    cache = (W, A_prev, b)
    return Z, cache


def l_forward_non_linear_activation(Z, a_func):
    """Computes and returns non linear activation A of layer along with caches (Z).
    :param Z: linear activation of layer
    :param a_func: non-linear activation function name
    :return A, caches (Z)
    """

    if a_func == "relu":
        A, cache = relu(Z)
    elif a_func == "sigmoid":
        A, cache = sigmoid(Z)
    else:
        A, cache = relu(Z)

    return A, cache


def l_fprop(W, A_prev, b, a_func):
    """
    Computes and returns the activation of the layer.
    :param A_prev: activations from  previous layer
    :param W: weights of the layer
    :param b: biases of the layer
    :param a_func: activation function name
    :return: A (computed activations) and cache ((W, A_prev, b), (Z))
    """

    Z, linear_cache = l_forward_linear_activation(W, A_prev, b)
    A, activation_cache = l_forward_non_linear_activation(Z, a_func)
    cache = (linear_cache, activation_cache)
    return A, cache


def nn_fprop(X, params):
    """
    Computes the activations of the model and returns them along with a collection of caches (As, Zs, Ws, Bs).
    :param X: activation of the input layer
    :param params: parameters of the model
    :return: AL (output layer activations) and caches ((Z), (W, A_prev, b))
    """

    caches = []
    A = X
    L = len(params) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = l_fprop(params["W" + str(l)], A_prev, params["b" + str(l)], "relu")
        caches.append(cache)

    AL, cache = l_fprop(params["W" + str(L)], A, params["b" + str(L)], "sigmoid")
    caches.append(cache)
    assert (AL.shape == (1, X.shape[1]))
    return AL, caches


def nn_loss(AL, Y):
    """
    Computes the loss of the model.
    :param AL: activations of output layer
    :param Y: training set of labels
    :return: cost of model
    """

    loss = -(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    return loss


def nn_cost(AL, Y):
    """
    Computes the cost of the model.
    :param AL: activations of output layer
    :param Y: training set of labels
    :return: cost of model
    """

    m = Y.shape[1]
    loss = nn_loss(AL, Y)
    cost = (1.0 / m) * np.sum(loss)
    cost = np.squeeze(cost)
    return cost


def relu_backward(dA, cache):
    """
    Computes and returns dZ (the gradient of Z) of layer with respect to the sigmoid activation function.
    :param dA: the activations of layer
    :param cache: cache of the layer (Z)
    :return: dZ
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0  # When z <= 0, you should set dz to 0 as well.
    assert (dZ.shape == Z.shape)
    return dZ


def sigmoid_backward(dA, cache):
    """
    Computes and returns dZ (the gradient of Z) of layer with respect to the sigmoid activation function.
    :param dA: the activations of layer
    :param cache: cache of the layer (Z)
    :return: dZ
    """

    Z = cache
    s = 1.0 / (1.0 + np.exp(-Z))
    dZ = dA * s * (1.0 - s)
    assert (dZ.shape == Z.shape)
    return dZ


def l_backward_linear_activation(dZ, cache):
    """
    Computes and returns the gradients dA_prev, dW, and db of layer.
    :param dZ: gradient of Z of layer
    :param cache: cache of layer (W, A_prev, b)
    :return: dA_prev, dW, and db
    """

    W, A_prev, b = cache
    m = A_prev.shape[1]
    dW = (1.0 / m) * np.dot(dZ, A_prev.T)
    db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db


def l_backward_non_linear_activation(dA, cache, a_func):
    """
    Computes and returns the gradient dZ of layer
    :param dA: activations of layer
    :param cache: activation cache (W, A_prev, b)
    :param a_func: activation function name
    :return: dZ
    """
    if a_func == "relu":
        dZ = relu_backward(dA, cache)

    elif a_func == "sigmoid":
        dZ = sigmoid_backward(dA, cache)

    else:
        dZ = relu_backward(dA, cache)

    return dZ


def l_bprop(dA, cache, a_func):
    """
    Computes and returns the gradients of the layer.
    :param dA: the activations of layer
    :param cache: cache of layer ((W, A_prev, b), (Z))
    :param a_func: activation function name
    :return: dA_prev, dW, db
    """

    linear_cache, activation_cache = cache
    dZ = l_backward_non_linear_activation(dA, activation_cache, a_func)
    dA_prev, dW, db = l_backward_linear_activation(dZ, linear_cache)
    return dA_prev, dW, db


def nn_bprop(AL, Y, caches):
    """
    Computes and returns the gradients of the model's parameters.
    :param AL: activations of the output layer
    :param Y: training set of labels
    :param caches: values used to compute activations of all layers ((Z), (W, A_prev, b))
    :return: gradients
    """

    # Prepare compute variables.
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Compute gradients of output layer.
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = l_bprop(dAL, current_cache, "sigmoid")

    # Compute gradients of all previous layers.
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = l_bprop(grads["dA" + str(l + 1)],
                                                                                           current_cache, "relu")

    return grads


def nn_params_update(params, l_rate, grads):
    """
    Updates model's parameters after one step of gradient descent.
    :param params: model's parameters
    :param l_rate: model's learning rate
    :param grads: parameter's  gradients
    :return: updated parameters
    """

    L = len(params) // 2
    for l in range(L):
        params["W" + str(l + 1)] = params["W" + str(l + 1)] - (l_rate * grads["dW" + str(l + 1)])
        params["b" + str(l + 1)] = params["b" + str(l + 1)] - (l_rate * grads["db" + str(l + 1)])

    return params


def nn_train(X, Y, h_params, params):
    """
    Trains the model via optimizing the parameters via gradient descent.
    :param X: training set of input feature vectors
    :param Y: training set of labels
    :param h_params: hyper parameters of model
    :param params: parameters of model
    :return: model (updated params W and b of each layer)
    """

    costs = []
    for i in range(0, h_params["n_iterations"]):

        # Update parameters.
        AL, caches = nn_fprop(X, params)
        grads = nn_bprop(AL, Y, caches)
        params = nn_params_update(params, h_params["l_rate"], grads)

        # Accumulate cost.
        cost = nn_cost(AL, Y)
        p_cost = h_params["print_cost"]
        if p_cost and i % 100 == 0 or i == h_params["n_iterations"] - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == h_params["n_iterations"]:
            costs.append(cost)

    model = params
    return model


def nn_model(dataset, h_params, params):
    """
    Builds a model.
    :param dataset: training and testing sets for model including classes and other
    values related to the data sets.
    :param h_params: hyper parameters of the model
    :param params: parameters of the model
    :return: the model
    """

    model = nn_train(dataset["X_train"], dataset["Y_train"], h_params, params)
    nn_test(dataset, model)
    return model


def predict(X, Y, params):
    """
    Returns Y (predicted label set) by predicting each label for each input feature vector of the set X.
    :param X: set of input feature vectors to label
    :param Y: set of labels labeling X
    :param params: set of input feature vectors
    :return: predicted label set Y
    """

    m = X.shape[1]  # number of training examples
    n = len(params) // 2  # number of layers in the neural network
    Y_prediction_labels = np.zeros((1, m))

    # Forward propagation
    Y_prediction_probabilities, caches = nn_fprop(X, params)

    # Convert probabilities to binary classification {1, 0} or {true, false}
    for i in range(0, Y_prediction_probabilities.shape[1]):
        if Y_prediction_probabilities[0, i] > 0.5:
            Y_prediction_labels[0, i] = 1
        else:
            Y_prediction_labels[0, i] = 0

    # Print accuracy.
    # print("\nAccuracy: " + str(np.sum((Y_prediction_labels == Y) / m) * 100) + str(" %"))

    return Y_prediction_labels


def nn_test(dataset, model):
    # Collect model predictions.
    Y_prediction_test = predict(dataset["X_test"], dataset["Y_test"], model)
    Y_prediction_train = predict(dataset["X_train"], dataset["Y_train"], model)

    # Print training and test accuracy.
    print("\nTrain accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - dataset["Y_train"])) * 100))
    print("Test accuracy: {} %\n".format(100 - np.mean(np.abs(Y_prediction_test - dataset["Y_test"])) * 100))


def nn_predict_user_input(model):
    """
    User is prompted for an image and the model predicts whether or not the image is a cat or non cat picture.
    :param model: the model for recognizing cat or non cat pictures
    :return: 0 if image is a non cat picture or 1 if image is a cat picture
    """
    return 0


def main():
    """
    Runs the program.
    :return: 0 if exit success otherwise failure
    """

    # Train model and predict if user's image is a cat or non-cat picture.
    dataset = dataset_load()
    h_params = nn_h_params_init()
    params = nn_params_init(h_params["l_dims"])
    model = nn_model(dataset, h_params, params)
    prediction = nn_predict_user_input(model)
    return 0


if __name__ == '__main__':
    """Calls main."""
    main()
