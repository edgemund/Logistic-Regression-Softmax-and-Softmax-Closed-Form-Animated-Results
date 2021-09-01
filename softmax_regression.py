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


class softmax_regression():
    def __init__(self, data, label, num_class):
        self.label = label
        self.num_class = num_class
        self.num_data = len(data)
        data = self.preprocess(data)
        bias = np.ones(len(data),)
        self.data = np.vstack((data.T, bias)).T
        self.num_feature = self.data.shape[1]
        self.theta = np.random.randn(self.num_feature, self.num_class)
        if num_class == 2:
            self.lr = 1
        elif num_class == 3:
            self.lr = 1.5
        self.color_list = ['r', 'g', 'b', 'y']

    def preprocess(self, data):
        data = data - np.mean(data, axis=0, keepdims=True)
        data = data / (np.max(data, axis=0, keepdims=True) -
                       np.min(data, axis=0, keepdims=True))
        return data

    def softmax(self, data):
        ''' Calculate Softmax Prediction Vector. '''
        # `Feature` is an n by num_class matrix
        feature = np.dot(data, self.theta)
        # Prevent overflow by subtracting the maximum vector
        feature -= np.max(feature, axis=1, keepdims=True)
        exp_feature = np.exp(feature)
        sum_exp_feature = np.sum(exp_feature, axis=1, keepdims=True)
        return exp_feature / sum_exp_feature

    def loss(self):
        ''' Use Log-likelihood estimation to compute loss function. '''
        score = np.dot(self.data, self.theta)
        score -= np.max(score, axis=1, keepdims=True)
        sum_exp_score = np.sum(np.exp(score), axis=1)
        loss = np.log(sum_exp_score)
        # Remove redundant terms: the correct term
        loss -= score[np.arange(self.num_data), self.label]
        loss = (1. / self.num_data) * np.sum(loss)
        return loss

    def update_parameter(self, stochastic=0):
        ''' Calculate Grediant. '''
        if stochastic != 0:
            rand_i = np.random.randint(0, self.num_data, stochastic)
            if stochastic == 1:
                self.lr = 10
                x = self.data[rand_i]
                y = self.label[rand_i]
            else:
                self.lr = 5
                x = self.data[rand_i]
                y = self.label[rand_i]
        else:
            x = self.data
            y = self.label
        softmax = self.softmax(x)
        softmax[np.arange(len(x)), y] -= 1.
        gred = (1. / self.num_data) * np.dot(x.T, softmax)
        self.theta -= self.lr * gred
        print('theta:\n', self.theta)


if __name__ == '__main__':
    cl = input('Input Classification number: ')
    opt = input('Input Optimization strategy''s number: ')
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

    softmax_reg = softmax_regression(data, label, int(cl))
    print('Initiated theta value is:\n', softmax_reg.theta)

    loss_list = []
    step_list = []
    acc_list = []
    plt.ion()
    fig, ax = plt.subplots(1, 4, figsize=(16, 5))
    for steps in range(300):
        step_list.append(steps)
        pred = softmax_reg.softmax(softmax_reg.data)
        classification = np.argmax(pred, 1)
        loss = softmax_reg.loss()
        print('Current Loss is:\n', loss)
        loss_list.append(loss)

        plt.subplot(1, 4, 1)
        plt.title('Ground Truth')
        for i in range(int(cl)):
            data_x = np.array(data.T[0][label == i])
            data_y = np.array(data.T[1][label == i])
            plt.scatter(data_x, data_y, c=softmax_reg.color_list[i])

        plt.subplot(1, 4, 2)
        plt.title('Classification Plot')
        for i in range(int(cl)):
            data_x = np.array(data.T[0][classification == i])
            data_y = np.array(data.T[1][classification == i])
            if len(data_x) == 0:
                continue
            plt.scatter(data_x, data_y, c=softmax_reg.color_list[i])
        ax[1].cla()
        plt.subplot(1, 4, 3)
        plt.title('Loss')
        ax[2].cla()
        plt.plot(step_list, loss_list, c='b', ls='-', marker='o')
        plt.subplot(1, 4, 4)
        acc = sum(label == classification) / softmax_reg.num_data
        acc_list.append(acc)
        plt.plot(step_list, acc_list, c='g', ls='-', marker='*')
        plt.title('Accuracy')
        plt.pause(0.1)
        if opt == 0:
            softmax_reg.update_parameter()
        else:
            softmax_reg.update_parameter(int(opt))
