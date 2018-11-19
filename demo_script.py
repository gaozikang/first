#!/usr/bin/python

import Network
import numpy as np
import itertools

print("开始训练!")
# 读取数据
T, total_population = Network.get_data('/home/zikang/文档/population_prediction/Neural Networks/population_prediction.xlsx')
predict = [21]
# 数据预处理
m = 0; n = 20; epochs = 1000
training_inputs = np.array(T[m:n], dtype = float).reshape([len(T[m:n]), 1]) #将 T 转化为数组
training_results = np.array(total_population[m:n], dtype = float).reshape([len(total_population[m:n]), 1])
Predicted_T = np.array(predict, dtype=float).reshape([len(predict), 1])
# scale units
training_inputs = training_inputs / np.amax(training_inputs, axis=0)  # maximum of training_inputs array
training_results = training_results / 10
Predicted_T = Predicted_T / np.amax(Predicted_T,axis=0)  # maximum of Predicted_T (our input data for the prediction)
NN = Network.Neural_Network([1, 3, 1], 0.01)
mse = []
for i in range(epochs):  # trains the NN epochs times
    print(" #" + str(i) + "\n")
    square_error = []
    for single_inputs, single_results in zip(training_inputs, training_results):
        single_inputs = np.array([single_inputs])
        single_results = np.array([single_results])
        print("Input (scaled): \n", single_inputs)
        print("Actual Output: \n", single_results)
        NN.train(single_inputs, single_results)
        print("Predicted Output: \n", NN.feedforward(single_inputs))
        print("Loss: \n", np.square(single_results - NN.feedforward(single_inputs)))
        print("\n")
        square_error.append(np.square(single_results - NN.feedforward(single_inputs)))
    mse.append(np.mean(square_error, axis = 0).tolist()) # mean sum square loss
# Network.saveWeights()
NN.predict(Predicted_T)
mse = list(itertools.chain.from_iterable(mse))
mse = list(itertools.chain.from_iterable(mse))
# plot
Network.plot_loss(epochs, mse)
