############################################################################
# LOGISTIC REGRESSION                                                      #
# Note: NJUST Machine Learning Assignment.                                 #
# Optimization: Grediant Descent (GD), Stochastic Grediant Descent(SGD).   #
# Author: Edmund Sowah                                                     #
############################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class logistic_regression():
    def __init__(self, data, label):
        self.label = label.reshape(-1, 1)
        self.num_data = len(data)
        bias = np.ones(len(data),)
        data = self.preprocess(data)
        self.data = np.vstack((data.T, bias)).T
        self.theta = np.random.randn(self.data.shape[1], 1)
        self.lr = 0.8
        self.color_list = ['r', 'g', 'b', 'y']

    def preprocess(self, data):
        data = data - np.mean(data, axis=0, keepdims=True)
        data = data / (np.max(data, axis=0, keepdims=True) -
                       np.min(data, axis=0, keepdims=True))
        return data

    def sigmoid(self, score):
        return 1. / (1. + np.exp(-score))

    def hypothesis(self, data):
        ''' Calculate Classification Hypothesis. '''
        score = data.dot(self.theta)
        hypothesis = self.sigmoid(score)
        return hypothesis

    def loss(self):
        ''' Using Log-Likelihood estimation to compute loss function. '''
        hypothesis = self.hypothesis(self.data)
        loss = - np.sum(np.dot(self.label.T, np.log(hypothesis)) +
                        np.dot((1 - self.label).T, np.log(1 - hypothesis)))
        return loss / self.num_data

    def update_parameter(self, stochastic=0):
        ''' Calculate Grediant. '''
        if stochastic != 0:
            rand_i = np.random.randint(0, self.num_data, stochastic)
            if stochastic == 1:
                self.lr = 6
                x = self.data[rand_i]
                y = self.label[rand_i]
            else:
                self.lr = 3
                x = self.data[rand_i]
                y = self.label[rand_i]
        else:
            x = self.data
            y = self.label
        grad = np.dot(x.T, (self.hypothesis(x) - y)) / self.num_data
        self.theta -= self.lr * grad
        print('theta:\n', self.theta)


if __name__ == '__main__':
    opt = input('Input Optimization strategy number: ')
    # loading the data
    exam_data = np.loadtxt('Exam\exam_x.dat')
    exam_label = np.loadtxt('Exam\exam_y.dat', dtype=int)
    data = exam_data
    label = exam_label

    log_reg = logistic_regression(data, label)
    print('Initiated theta Value is:\n', log_reg.theta)

    loss_list = []
    step_list = []
    acc_list = []
    plt.ion()
    fig, ax = plt.subplots(1, 4, figsize=(16, 5))
    for steps in range(300):
        step_list.append(steps)
        pred = log_reg.hypothesis(log_reg.data).flatten()
        classification = pred > 0.5
        classification = np.array(classification, dtype=int)
        loss = log_reg.loss()
        print('Current loss value is:\n', loss)
        loss_list.append(loss)

        plt.subplot(1, 4, 1)
        plt.title('Ground Truth')
        for i in range(2):
            data_x = np.array(data.T[0][label == i])
            data_y = np.array(data.T[1][label == i])
            plt.scatter(data_x, data_y, c=log_reg.color_list[i])

        plt.subplot(1, 4, 2)
        plt.title('Classification Plot')
        for i in range(2):
            data_x = np.array(data.T[0][classification == i])
            data_y = np.array(data.T[1][classification == i])
            if len(data_x) == 0:
                continue
            plt.scatter(data_x, data_y, c=log_reg.color_list[i])
        ax[1].cla()
        plt.subplot(1, 4, 3)
        plt.title('Loss')
        ax[2].cla()
        plt.plot(step_list, loss_list, c='b', ls='-', marker='o')
        plt.subplot(1, 4, 4)
        acc = sum(label == classification) / log_reg.num_data
        acc_list.append(acc)
        plt.plot(step_list, acc_list, c='g', ls='-', marker='*')
        plt.title('Accuracy')
        plt.pause(0.1)
        if opt == '0':
            log_reg.update_parameter()
        else:
            log_reg.update_parameter(stochastic=int(opt))
