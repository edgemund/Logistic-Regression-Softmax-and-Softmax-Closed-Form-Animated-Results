############################################################################
# LOGISTIC REGRESSION                                                      #
# Note: NJUST Machine Learning Assignment.                                 #
# Task: Binary Classification, Multi-class Classification.                 #
# Optimization: Grediant Descent (GD), Stochastic Grediant Descent(SGD).   #
# Author: Edmund Sowah                                                     #
############################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


if __name__ == '__main__':
    cl = input('Input Classification number: ')
    if cl == '2':
        # Import Binary data
        exam_data = np.loadtxt('Exam\exam_x.dat')
        exam_label = np.loadtxt('Exam\exam_y.dat', dtype=int)
        data = exam_data
        label = exam_label
    elif cl == '3':
        # Import Multi-class data
        iris_data = np.loadtxt('Iris\iris_x.dat')
        iris_label = np.loadtxt('Iris\iris_y.dat', dtype=int)
        data = iris_data
        label = iris_label
    else:
        print('no {}-class task available!'.format(cl))

    data = np.vstack((data.T, np.ones(len(data)))).T
    color_list = ['r', 'g', 'b', 'y']
    y_mat = np.zeros((len(label), int(cl)))
    y_mat += 1e-32
    y_mat[np.arange(len(label)), label] = 1.
    y_mat = np.log(y_mat)
    inv_data = np.linalg.pinv(data)
    theta = np.dot(inv_data, y_mat)

    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    classification = np.dot(data, theta)
    classification = np.argmax(classification, 1)

    plt.subplot(1, 2, 1)
    plt.title('Ground Truth')
    for i in range(int(cl)):
        data_x = np.array(data.T[0][label == i])
        data_y = np.array(data.T[1][label == i])
        plt.scatter(data_x, data_y, c=color_list[i])

    plt.subplot(1, 2, 2)
    plt.title('Classification Plot')
    for i in range(int(cl)):
        data_x = np.array(data.T[0][classification == i])
        data_y = np.array(data.T[1][classification == i])
        if len(data_x) == 0:
            continue
        plt.scatter(data_x, data_y, c=color_list[i])
    plt.pause(100)
