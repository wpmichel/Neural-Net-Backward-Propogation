import math
import random
import matplotlib.pyplot as plt


"""
A unit of a neural network.
"""
class NNUnit:
    def __init__(self, activation, inputs = None, weights = None):
        self.activation = activation
        self.inputs = inputs or []
        self.weights = weights or []
        self.value = None


"""
Creates a neural network.
"""
def createNN(input_layer_size, hidden_layer_sizes, output_layer_size, activation):
    layer_sizes = [input_layer_size] + hidden_layer_sizes + [output_layer_size]
    network = [[NNUnit(activation) for _ in range(s)] for s in layer_sizes]
    dummy_node = NNUnit(activation)
    dummy_node.value = 1.0
    
    # create weighted links
    for layer_idx in range(1, len(layer_sizes)):
        for node in network[layer_idx]:
            node.inputs.append(dummy_node)
            node.weights.append(0.0)
            for input_node in network[layer_idx - 1]:
                node.inputs.append(input_node)
                node.weights.append(0.0)
    
    return network

""" Sigmoid activation function """
def sigmoid(z):
    return float(1.0/(1.0 + math.exp(-1.0*float(z))))

""" Sigmoid activation function derivative """
def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))
print(sigmoid_deriv(0))


"""
[findInJ(n,idx)] is the InJ value for node [n] of layer [idx].
"""
def findInJ(n,idx): 
    total = 0
    for idx,w in enumerate(n.weights):
        input_n = n.inputs[idx]
        if idx == 1: 
            total += w*(input_n.value)
        else:
            total += w*(input_n.activation(input_n.value))
    return total


"""
[back_prop_learning(examples, network, epochs, learning_rate, deriv = sigmoid_deriv)] is the trained network and 
mean squared errors resulting from backward propograted learning performed on neural network [network] with [examples],
[learning_rate] and derivation function [deriv] over [epochs] runs for each individual example.
Returns: network,errors
"""
def back_prop_learning(examples, network, epochs, learning_rate, deriv = sigmoid_deriv):
    """
    [findDeltaI(n, deltas)] is the delta value for node [n] according to existing deltavalues [deltas]
    """   
    def findDeltaI(n, deltas):
        total = 0
        for d_node in deltas: 
            for i in range(len(d_node.inputs)): 
                if d_node.inputs[i] == n:
                    total += d_node.weights[i]*deltas[d_node]
        return deriv(n.value)*total
    """
    [mse(output_layer,example)] is the mean squared error according to example output [example] and output 
    layer [output_layer.]
    """
    def mse(output_layer, example):
        total = 0
        for idx,e in enumerate(example):
            n = output_layer[idx]
            total += (e-n.activation(n.value))**2
        return float((1.0/len(example))*total)
    
    # Randomly initialize weights
    for l in network:
        for n in l: 
            n.weights = [random.uniform(-0.5,0.5) for i in range(len(n.weights))]
            
    errors = []
    for t in range(epochs):
        epoch_error = 0
        for e in examples:
            deltas = {}
            inputs = e[0]
            outputs = e[1]
            for idx,n in enumerate(network[0]): # set values for input layer
                n.value = inputs[idx]
                
            for l_idx,layer in enumerate(network[1:]): # find inj values
                for n in layer:
                    n.value = findInJ(n,l_idx)
                    
            # Propogate backwards
            for idx,rev_layer in enumerate(network[::-1]):
                if not deltas: # output layer
                    for idx,n in enumerate(rev_layer):
                        deltas[n] = deriv(n.value)*(outputs[idx]-(n.activation(n.value)))
                else:
                    # L-1 to 1
                    for n in rev_layer: 
                        delta = findDeltaI(n,deltas)
                        deltas[n] = delta
                        
            # Update weights
            for idx, layer in enumerate(network[1:]):
                for n in layer: 
                    for i_idx, i in enumerate(n.inputs):
                        n.weights[i_idx] = n.weights[i_idx]+(learning_rate*i.activation(i.value)*deltas[n])
            
            epoch_error += mse(network[-1],outputs)
        errors.append(epoch_error)
    return network,errors

dataset = [([0,0],[0,0]), ([0,1],[0,1]), ([1,0],[0,1]), ([1,1],[1,0])]
network = createNN(2, [2,2], 2, sigmoid)
num_epochs = 10000
trained_network, total_errs = back_prop_learning(dataset, network, num_epochs, 0.25)
_, axis = plt.subplots()
epochs = [e for e in range(0, num_epochs, 1)]
axis.plot(epochs, total_errs)
axis.set(xlabel = 'Number of Epochs', ylabel = 'Total Error on Training Set')
axis.grid()
plt.show()


"""
[predict(network,exmaple)] is a list of outputted prediction values from propogating the inputs of [example] 
through the neural netwrok [network].
"""
def predict(network, example):
    for idx,n in enumerate(network[0]): 
        n.value = example[idx]

    for l_idx,layer in enumerate(network[1:]): 
        for n in layer:
            n.value = findInJ(n,l_idx)
    return [n.activation(n.value) for n in network[-1]]


test_set = [[0,0], [0,1], [1,0], [1,1]]
correct_out = [[0,0], [0,1], [0,1], [1,0]]
for t in range(len(test_set)):
    print('Example =', test_set[t])
    preds = predict(trained_network, test_set[t])
    print('Prediction =', preds)
    print('Actual =', correct_out[t])
    devs = [abs(preds[p] - correct_out[t][p]) for p in range(len(preds))]
    print('Deviations =', devs)
    print()


data_set = [([p,q,r,s],[int(p and not r), int(q <= p)]) for p in [0,1] for q in [0,1] for r in [0,1] for s in [0,1]]
random.shuffle(data_set)
training_set = data_set[0:12]
network = createNN(4, [8,8], 2, sigmoid)
num_epochs = 5000
trained_network, total_errs = back_prop_learning(training_set, network, num_epochs, 0.1)
_, axis = plt.subplots()
epochs = [e for e in range(0, num_epochs, 1)]
axis.plot(epochs, total_errs)
axis.set(xlabel = 'Number of Epochs', ylabel = 'Total Error on Training Set')
axis.grid()
plt.show()
test_set = [x for (x,y) in data_set[12:]]
correct_out = [y for (x,y) in data_set[12:]]
num_misses = 0
for t in range(len(test_set)):
    preds = predict(trained_network, test_set[t])
    devs = [abs(preds[p] - correct_out[t][p]) for p in range(len(preds))]
    for d in devs:
        if d >= 0.1:
            num_misses += 1
print('Error Rate:', num_misses / 8.0 * 100.0, '%')


# Alternative Hyperparameter / Activation Functions

def sin(z):
    return math.sin(z)


def sin_deriv(z):
    return math.cos(z)


def tanh(z):
    return (math.exp(z)-math.exp(-1.0*z))/(math.exp(z)+math.exp(-1.0*z))


def tanh_deriv(z):
    return 1 - (tanh(z)**2)


def elu(z, alpha = 0.01):
    return 0.0*(math.exp(z)-1) if z <= 0 else z


def elu_deriv(z, alpha = 0.01):
    return alpha*math.exp(z) if z <= 0 else z


def relu(z): # subgradient
    return z if z > 0 else 0 


def relu_deriv(z): # subgradient
    return 1 if z > 0 else 0

training_set = [([0,0],[0,0]), ([0,1],[0,1]), ([1,0],[0,1]), ([1,1],[1,0])]
learning_rate = 0.25
num_epochs = 10000
funs = [sigmoid, sin, tanh, elu, relu]
drvs = [sigmoid_deriv, sin_deriv, tanh_deriv, elu_deriv, relu_deriv]
errs = []
r = random.getstate()
for f in range(len(funs)):
    network = createNN(2, [2,2], 2, funs[f])
    random.setstate(r)
    _, total_errs = back_prop_learning(training_set, network, num_epochs, learning_rate, deriv = drvs[f])
    errs.append(total_errs)
fig, axs = plt.subplots(5, 1, sharex = True, figsize = (7, 7))
fig.subplots_adjust(hspace = 0.25)
t = [i for i in range(0, num_epochs, 1)]
axs[0].plot(t, errs[0], color = 'b', label = 'sigmoid')
axs[1].plot(t, errs[1], color = 'g', label = 'sin')
axs[2].plot(t, errs[2], color = 'r', label = 'tanh')
axs[3].plot(t, errs[3], color = 'y', label = 'elu')
axs[4].plot(t, errs[4], color = 'm', label = 'relu')
axs[0].set(title = 'Total Error Over Time')
axs[4].set(xlabel = 'Epoch')
for i in range(len(axs)):
    axs[i].legend(loc = 'upper right')
plt.show()

"""
[check_alphas(examples, network, epochs, min_alpha, max_alpha, step_size, deriv = sigmoid_deriv)] is a list of errors 
corresponding to backward propogation learning on the given [network] and [exmaples] with 
the learning rate values inclusively between [min_alpha] and [max_alpha] with step size [step_size]. The error in 
each entry is the Mean Squared Error calculated in the final epoch, or, in other words, the MSE of epoch [epochs].

"""
def check_alphas(examples, network, epochs, min_alpha, max_alpha, step_size, deriv = sigmoid_deriv):
    alphas=[]
    a = min_alpha
    while a <= max_alpha:
        n,error = back_prop_learning(examples, network, epochs, a, deriv)
        alphas.append(error[-1])
        a += step_size
    return alphas
    
dataset = [([0,0],[0,0]), ([0,1],[0,1]), ([1,0],[0,1]), ([1,1],[1,0])]
network = createNN(2, [2,2], 2, sigmoid)
num_epochs = 5000
mina = -0.3
maxa = 0.3
ss = 0.05
errs = check_alphas(dataset, network, num_epochs, mina, maxa, ss, sigmoid_deriv)
_, axis = plt.subplots()
alphas = []
while mina <= maxa:
    alphas.append(mina)
    mina += ss
axis.plot(alphas, errs)
axis.set(xlabel = 'Learning Rates', ylabel = 'Total Error at Final Epoch')
axis.grid()
plt.show()
