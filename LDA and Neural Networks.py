
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 19:23:36 2017

@author: Virinchi
"""
#Question1:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
data = pd.read_csv("C:\\Users\\virinchi\\Desktop\\SCLC_study_output_filtered_2.csv")
data = pd.read_csv(io.StringIO(u""+data.to_csv(header=None,index=False)), header=None)

#converting the catergorical variables to numerical
data.iloc[0:20:,0] = 0
data.iloc[20:41:,0] = 1
label_dict = {0: 'NSCLC', 1: 'SCLC'}

#Splitting the data set into independent and target variable datasets
x_train = data.iloc[:,1:20].values
y_train = data.iloc[:,0].values

#calculating the mean vectors
mean_vectors = []
for classes in range(0,2):
    mean_vectors.append(np.mean(x_train[y_train==classes], axis=0))

#Calculating within class scatter matrix
Within_Scatter = np.zeros((19,19))
for cl,mv in zip(range(0,3), mean_vectors):
    class_sc_mat = np.zeros((19,19))                  # scatter matrix for every class
    for row in x_train[y_train == cl]:
        row, mv = row.reshape(19,1), mv.reshape(19,1) # make column vectors
        class_sc_mat += (row-mv).dot((row-mv).T)
    Within_Scatter += class_sc_mat                             # sum class scatter matrices

#Calculating Between class scatter matrix
overall_mean = np.mean(x_train, axis=0)
Between_Scatter = np.zeros((19,19))
for i,mean_vec in enumerate(mean_vectors):  
    n = x_train[y_train==i+1,:].shape[0]
    mean_vec = mean_vec.reshape(19,1) # make column vector
    overall_mean = overall_mean.reshape(19,1) # make column vector
    Between_Scatter += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

#Calculating Eigen vectors and eigen values
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Within_Scatter).dot(Between_Scatter))
for i in range(len(eig_vals)):
    eigv = eig_vecs[:,i].reshape(19,1)
    np.testing.assert_array_almost_equal(np.linalg.inv(Within_Scatter).dot(Between_Scatter).dot(eigv),
                                         eig_vals[i] * eigv,
                                         decimal=6, err_msg='', verbose=True)

# Make a list of (eigenvalue, eigenvector) tuples
eigen_val_and_vec_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_val_and_vec_pairs = sorted(eigen_val_and_vec_pairs, key=lambda k: k[0], reverse=True)

#checking the variance explained by LDA axis    
print('Variance explained by different eigen vectors:\n')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eigen_val_and_vec_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))

#we can see that the variance explained by LD1 axis is 100%
W = np.hstack((eigen_val_and_vec_pairs[0][1].reshape(19,1)))
print('Matrix W:\n', W.real)

#Implementing Sklearn LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
x_train = data.iloc[:,1:20].values
y_train = np.asarray(data.iloc[:,0], dtype = "|S6")

# LDA from SKlearn
sklearn_lda = LDA()
X_lda_sklearn = sklearn_lda.fit(x_train, y_train)
x = (X_lda_sklearn.coef_)
my_list=[]
for i in x:
    my_list.append((i))
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
print("Percentage of variance explained by First LD1 is:",(X_lda_sklearn.explained_variance_ratio_)*100)
print("W matrix from Sklearn LDA is:",my_list)
import numpy as np

val = 0. # this is the value where you want the data to appear on the y-axis.
ar = W # just as an example array
ar1 = x
plt.plot(ar, np.zeros_like(ar) + val, 'x')
plt.title("Plot for W matrix from My function")
plt.show()
plt.plot(ar1, np.zeros_like(ar1) +val, 'o')
plt.title("Plot for W matrix from Sklearn Library")
plt.show()

#************************************************************************************************************************************************************************
#****************************************************************************************************************************************

#question 2:
#Neural Networks
from colorama import Fore, Back, Style

print(Fore.RED +"\n Neural Networks plots")
from math import exp
from random import seed
import numpy as np

# Initialize a network with randomly distributed uniform weights
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[np.random.uniform(0.0,1.0) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[np.random.uniform(0.0,1.0) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i !=0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']
            
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            sum_error += sum(([0.5*((expected[i]-outputs[i])**2) for i in range(len(expected))]))
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        #Appending the weights of individual parameters to the lists
        theta11.append(network[0][0]['weights'][0])
        theta21.append(network[0][0]['weights'][1])
        theta31.append(network[0][0]['weights'][2])
        theta12.append(network[0][1]['weights'][0])
        theta22.append(network[0][1]['weights'][1])
        theta32.append(network[0][1]['weights'][2])
        theta41.append(network[1][0]['weights'][0])
        theta51.append(network[1][0]['weights'][1])
        theta61.append(network[1][0]['weights'][2])
        theta42.append(network[1][1]['weights'][0])
        theta52.append(network[1][1]['weights'][1])
        theta62.append(network[1][1]['weights'][2])
        #print('>Iteration=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        iteration_list.append(epoch)
        error_list.append(sum_error)

import matplotlib.pyplot as plt
seed(1)
dataset = [[0.05,0.1,0.01,0.99]]
n_inputs = 2
n_outputs = 2
iteration_list = []
error_list = []
theta11 = []
theta12 = []
theta21 = []
theta22 = []
theta31 = []
theta32 = []
theta41 = []
theta42 = []
theta51 = []
theta52 = []
theta61 = []
theta62 = []
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 700, n_outputs)

#plotting the scatter plots 
plt.scatter(x= iteration_list, y = error_list)
plt.title("Plot between iterations and total cost")
plt.xlabel("Iterations")
plt.ylabel("Total cost")
plt.show()
plt.scatter(x= iteration_list, y = theta11)
plt.title("Plot between iterations and parameters of bias node to hidden layer first node")
plt.xlabel("Iterations")
plt.ylabel("parameters of bias node")
plt.show()
plt.scatter(x= iteration_list, y = theta21)
plt.title("Plot between iterations and parameters of first input node to hidden layer first node")
plt.xlabel("Iterations")
plt.ylabel("parameters of first input node")
plt.show()
plt.scatter(x= iteration_list, y = theta31)
plt.title("Plot between iterations and parameters of second input node to hidden layer first node")
plt.xlabel("Iterations")
plt.ylabel("parameters of second input node")
plt.show()
plt.scatter(x= iteration_list, y = theta12)
plt.title("Plot between iterations and parameters of bias node to hidden layer second node")
plt.xlabel("Iterations")
plt.ylabel("parameters of bias node")
plt.show()
plt.scatter(x= iteration_list, y = theta22)
plt.title("Plot between iterations and parameters of first input node to hidden layer second node")
plt.xlabel("Iterations")
plt.ylabel("parameters of first input node")
plt.show()
plt.scatter(x= iteration_list, y = theta32)
plt.title("Plot between iterations and parameters of second input node to hidden layer second node")
plt.xlabel("Iterations")
plt.ylabel("parameters of second input node")
plt.show()
plt.scatter(x= iteration_list, y = theta41)
plt.title("Plot between iterations and parameters of hidden layer bias node to first output node")
plt.xlabel("Iterations")
plt.ylabel("parameters of bias node")
plt.show()
plt.scatter(x= iteration_list, y = theta51)
plt.title("Plot between iterations and parameters of first hidden node to first output node")
plt.xlabel("Iterations")
plt.ylabel("parameters of first hidden node")
plt.show()
plt.scatter(x= iteration_list, y = theta61)
plt.title("Plot between iterations and parameters of second hidden node to first output node")
plt.xlabel("Iterations")
plt.ylabel("parameters of second hidden node")
plt.show()
plt.scatter(x= iteration_list, y = theta42)
plt.title("Plot between iterations and parameters of hidden layer bias node to second output node")
plt.xlabel("Iterations")
plt.ylabel("parameters of bias node")
plt.show()
plt.scatter(x= iteration_list, y = theta52)
plt.title("Plot between iterations and parameters of first hidden node to second output node")
plt.xlabel("Iterations")
plt.ylabel("parameters of first hidden node")
plt.show()
plt.scatter(x= iteration_list, y = theta62)
plt.title("Plot between iterations and parameters of second hidden node to second output node")
plt.xlabel("Iterations")
plt.ylabel("parameters of second hidden node")
plt.show()