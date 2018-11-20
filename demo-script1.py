#!/usr/bin/python

import Network
import numpy as np
import itertools

print("开始训练!")
# 读取数据
T, total_population = Network.get_data('/home/zikang/文档/population_prediction/Neural Networks/population_prediction.xlsx')
# 数据预处理
T = (T - np.min(T)) / (np.max(T) - np.min(T)) # 特征缩放
training_inputs = np.array(T[0:25], dtype = float).reshape([len(T[0:25]), 1]) #将 T 转化为数组
training_results = np.array(total_population[0:25], dtype = float).reshape([len(total_population[0:25]), 1])
test_inputs = np.array(T[25:30], dtype = float).reshape([len(T[25:30]), 1])
test_results = np.array(total_population[25:30], dtype = float).reshape([len(total_population[25:30]), 1])
Predicted_inputs = np.array(T[30:42], dtype = float).reshape([len(T[30:42]), 1])
training_results = training_results / 10
test_results = test_results / 10
print(test_inputs)
NN = Network.Neural_Network([1, 6, 1], 0.1)
mse = []; epochs = 100
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
mse = list(itertools.chain.from_iterable(mse))
mse = list(itertools.chain.from_iterable(mse))

# 测试
print("测试结果!")
for single_test_inputs in test_inputs:
    NN.predict(np.array([single_test_inputs]))

# 预测
print("预测结果!")
for single_Predicted_inputs in Predicted_inputs:
    NN.predict(np.array([single_Predicted_inputs]))
# plot
Network.plot_loss(epochs, mse)
