import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X / np.amax(X, axis=0)  # Normalize input
y = y / 100  # Normalize output
# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)
# Variable initialization
epoch = 5  # Setting training iterations
lr = 0.1  # Setting learning rate
inputlayer_neurons = 2  # Number of features in dataset
hiddenlayer_neurons = 3  # Number of hidden layer neurons
output_neurons = 1  # Number of neurons at output layer
# Weight and bias initialization
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))
# Training process
for i in range(epoch):
    # Forward Propagation
    hinp1 = np.dot(X, wh)
    hinp = hinp1 + bh
    hlayer_act = sigmoid(hinp)
    outinp1 = np.dot(hlayer_act, wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)
    # Backpropagation
    EO = y - output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad
    EH = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad
    wout += hlayer_act.T.dot(d_output) * lr
    wh += X.T.dot(d_hiddenlayer) * lr
    print("-----------Epoch-", i + 1, "Starts----------")
    print("Input: \n" + str(X))
    print("Actual Output: \n" + str(y))
    print("Predicted Output: \n", output)
    print("-----------Epoch-", i + 1, "Ends----------\n")
print("Final Predicted Output: \n", output)
