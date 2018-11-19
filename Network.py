#!/usr/bin/python

"""
N_Network.py
~~~~~~~~~~
Created on 2018-11-18
Author: gaozikang
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to get data
def get_data(file_name):
    data = pd.read_excel(file_name)
    X_parameter = []; Y_parameter = []
    for single_T ,single_total_population in zip(data['T'],data['Total_Population']):
        X_parameter.append(single_T)
        Y_parameter.append(single_total_population)
    return X_parameter,Y_parameter

class Neural_Network(object):

    def __init__(self, sizes, eta):
        """
        :param sizes: list类型，储存每层神经网络的神经元数目
                      譬如说：sizes = [2, 3, 2] 表示输入层有两个神经元、
                      隐藏层有3个神经元以及输出层有2个神经元
        """
        self.num_layers = len(sizes) # 有几层神经网络
        self.sizes = sizes # 储存各层神经元个数的列表
        self.eta = eta # 学习率
        # 初始化参数 W,b
        # 随机产生每条连接线的 weight 值（0 - 1）
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] # 3x1 1x3
        # 除去输入层，随机产生每层中 y 个神经元的 biase 值（0 - 1）
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # 3x1 1x1

    def feedforward(self, a):
        """
        前向传输计算每个神经元的值
        :param a: 输入值
        :return: 计算后每个神经元的值
        """
        for w, b in zip(self.weights, self.biases):
            # 加权求和以及加上 biase
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def backprop(self, x, y):
        """
        BP 算法
        """
        # 根据 biases 和 weights 的行列数创建对应的全部元素值为 0 的空矩阵
        nabla_w = [np.zeros(w.shape) for w in self.weights] # 3x1 1x3
        nabla_b = [np.zeros(b.shape) for b in self.biases] # 3x1 1x1
        # 前向传输
        activation = x
        # 储存每层的神经元的值的矩阵，下面循环会 append 每层的神经元的值
        activations = [x]
        # 储存每个未经过 sigmoid 计算的神经元的值
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # 反向过程
        # 求 δ 的值
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        # 乘于前一层的输出值
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            # 从倒数第 **l** 层开始更新，**-l** 是 python 中特有的语法表示从倒数第 l 层开始计算
            # 下面这里利用 **l+1** 层的 δ 值来计算 **l** 的 δ 值
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            nabla_b[-l] = delta
        self.weights = [w - self.eta * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - self.eta * nb for b, nb in zip(self.biases, nabla_b)]
        return (self.weights, self.biases)

    def sigmoid(self, z):
        """
        求 sigmoid 函数的值
        """
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        """
        求 sigmoid 函数的导数
        """
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def cost_derivative(self, output_activations, y):
        """
        二次损失函数
        """
        return (output_activations-y)

    def train(self, X, y):
        """
        训练 BP 算法
        """
        self.backprop(X, y)

    def predict(self, X):
        """
        利用训练得到 weights 和 biases 进行预测
        """
        print("Predicted data based on trained weights and biases: ")
        print("Predict Input(scaled): \n", X)
        print("Predict Output: \n", self.feedforward(X))

def plot_loss(x, y):
    """
    绘制损失函数随迭代次数变化图
    """
    fig = plt.subplots(1, 1)
    plt.plot(range(x), y, '-.r', lw=2, label='loss function')
    # plt.ylim((0, 1))
    # plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('epochs', fontsize=15)
    plt.ylabel('loss', fontsize=15)
    plt.title('The Process of Loss Change', fontsize=15)
    plt.tick_params(direction='in')
    plt.grid(linestyle = 'dotted')
    plt.legend(fancybox=True, loc='right')
    plt.show()
