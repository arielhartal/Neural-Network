import xxsubtype
from typing import List, Dict, Tuple

import numpy as np
import scipy.io as sio
from scipy.io import loadmat
import matplotlib.pyplot as plt


# Read and initialize data
mat_data1 = loadmat('GMMData.mat')
mat_data2 = loadmat('PeaksData.mat')
mat_data3 = loadmat('SwissRollData.mat')
C_train = mat_data1['Ct']
X_train = mat_data1['Yt']
labels = C_train.T
C_validation = mat_data1['Cv']
X_validation = mat_data1['Yv']



EPSILON = 1e-14
EARLY_STOPPING_EPSILON = 1e-4
KEEP_PROB = 0.8




# Question 1.1 softmax, loss and gradient:
def net_input(X, W):
    # If we use PeaksData change W to W.T
    return np.dot(W,X)


def softmax(z):
    return (np.exp(z) / np.sum(np.exp(z), axis=0))


def compute_loss(X, weights, labels):
    m = X.shape[1]
    # Compute softmax
    softmax_probs = softmax(net_input(X, weights))

    # Compute the loss
    loss = -np.sum(labels.T * np.log(softmax_probs)) / m

    return loss


def compute_gradient(X, smax, c):
    m = X.shape[1]  # Number of data points
    c_t = c.T
    gradient = []
    for p in range(c_t.shape[0]):
        gradient.append ((1 / m) * (X.dot((smax[p] - c_t[p]))))


    return gradient


# Question: 1.2 SGD:

def sgd_with_momentum(obj_func, grad_func, X, C, learning_rate=0.001, momentum=0.6, num_iterations=10):
    weights = np.random.randn(X.shape[0], C.shape[1])
    velocity = np.zeros_like(weights)
    losses = []
    for i in range(num_iterations):
        # Compute objective function and gradient
        loss = obj_func(X, weights, C)
        gradient = grad_func(X,smax, C)
        # Update weights using SGD with momentum
        velocity = np.array(velocity)
        gradient = np.array(gradient)
        velocity = momentum * velocity + learning_rate * gradient
        weights -= velocity
        losses.append(loss)

    return weights, losses



def sgd_with_momentum_least_squares(obj_func, grad_func, X, y, learning_rate=0.02, momentum=0.1, num_iterations=200):
    weights = np.random.randn(X.shape[0], y.shape[1])
    velocity = np.zeros_like(weights)
    losses = []
    for i in range(num_iterations):
        # Compute objective function and gradient
        loss = obj_func(X, weights, y)
        gradient = grad_func(weights)
        # Update weights using SGD with momentum
        velocity = momentum * velocity + learning_rate * gradient
        weights -= velocity

        losses.append(loss)

    return weights, losses


def least_squares_loss(X, weights, y):
    predictions = np.dot(weights, X.T)
    return np.mean((y - predictions)**2)

def least_squares_gradient(weights):
    np.random.seed(0)
    X = np.random.rand(10, 1)
    y = 2 * X + 1 + 0.1 * np.random.randn(10, 1)  # true function is y = 2x + 1, with some noise
    predictions = np.dot(weights, X.T)
    return -2 * np.mean((y - predictions) * X, axis=1, keepdims=True)






# gradient test

def F(x): # if we check 2.2.3 this should not be used
    return compute_loss(X_train, x, labels)



def g_F(x):
    return compute_gradient(X_train, smax, labels)

def gradient_test(weights):
    x = X_train
    d = np.random.randn(weights.shape[0], weights.shape[1])
    epsilon = 0.1
    F0 = F(weights)
    y0 = np.zeros(8)
    y1 = np.zeros(8)

    print("k\terror order 1\t\terror order 2")
    for k in range(1, 9):
        epsk = epsilon * (0.5 ** k)
        weights = weights + epsk
        Fk = F(weights + epsk * d)
        # If we use PeaksData change d to d.T
        F1 = F0 + epsk * np.sum(g_F(x) * d)
        y0[k - 1] = np.abs(Fk - F0)
        y1[k - 1] = np.abs(Fk - F1)
        print(k, "\t", np.abs(Fk - F0), "\t", np.abs(Fk - F1))

    plt.semilogy(range(1, 9), y0, label="Zero order approx")
    plt.semilogy(range(1, 9), y1, label="First order approx")
    plt.legend()
    plt.title("Successful Grad test in semilogarithmic plot")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.show()






def print_least_squares(losses):
    # Plot loss curve
    plt.plot(losses)
    plt.title('Loss over iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()









# Question 1.3

def train_sgd_with_momentum(X_train, C_train, X_validation, C_validation, weights, num_epochs=2, batch_size=10):
    # Initialize your model parameters


    # For storing accuracies
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        # Shuffle your training data
        shuffled_indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[shuffled_indices]
        C_train_shuffled = C_train[shuffled_indices]

        # Perform SGD with Momentum in mini-batches
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            C_batch = C_train_shuffled[i:i+batch_size]
            weights, _ = sgd_with_momentum(compute_loss, compute_gradient, X_batch, C_batch.T)

        # Compute accuracies
        train_acc = compute_accuracy(X_train, C_train, weights)
        val_acc = compute_accuracy(X_validation, C_validation, weights)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    return weights, train_accs, val_accs


def compute_accuracy(X, C, weights):
    # Compute the predictions
    predictions = np.dot(X.T, weights)
    # Apply the softmax function to get the probabilities
    probabilities = softmax(predictions)
    # Choose the class with the highest probability
    predicted_classes = np.argmax(probabilities, axis=1)
    # Compute the accuracy: the proportion of data points for which the predicted class is the same as the true class
    accuracy = np.mean(predicted_classes == C)
    return accuracy











# Running examples and checks:

# Question 1.1 - softmax and gradient test run example
weights = np.random.randn(X_train.shape[0], C_train.shape[0])
net_in = net_input(X_train, weights)
smax = (softmax(net_in))

def softmax_grad_test():
    gradient_test(weights)

#softmax_grad_test()

# Question 1.2 SGD run example:
# Least Squares Example
def sgd_least_squares_example():
    np.random.seed(0)
    X = np.random.rand(10, 1)
    y = 2 * X + 1 + 0.1 * np.random.randn(10, 1)  # true function is y = 2x + 1, with some noise
    weights_optimized, losses = sgd_with_momentum_least_squares(least_squares_loss, least_squares_gradient, X, y)
    print_least_squares(losses)

#sgd_least_squares_example()

# Question 1.3 SGD Softmax
def sgd_softmax():
    weights_optimized, train_accs, val_accs = train_sgd_with_momentum(X_train, C_train, X_validation, C_validation, weights)
    # Plot accuracies
    plt.plot(train_accs, label='Train loss')
    plt.plot(val_accs, label='Validation loss')
    plt.xlabel('Training Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()




# Question 1.3 - test
#sgd_softmax()



#Question 2.1 and 2.2

def init_net(layer_dims: List[int]) -> Dict[str, np.ndarray]:
    # Initialize empty dict to contain W and b
    parameters = {}
    # Loop over each layer
    for i in range(1, len(layer_dims)):
        # Initialize W using He initialization
        parameters[f"W{i}"] = np.random.randn(
            layer_dims[i], layer_dims[i - 1]) * np.sqrt(2 / layer_dims[i - 1])
        # Initialize b to constant zero
        parameters[f"b{i}"] = np.zeros((layer_dims[i], 1))
    return parameters




def tanh(Z):
    A = np.tanh(Z)
    activation_cache = {"Z": Z}
    return A, activation_cache

def tanh_derivative(Z):
    return 1 - np.square(np.tanh(Z))

def forward_lin(A: np.ndarray, W: np.ndarray,
                   b: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # Calculate linear component of the activation function
    Z = np.dot(W, A) + b
    return Z, {"A": A, "W": W, "b": b}


def softmax(Z: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # Subtract max for numerically stable exponential
    exp_max = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    # Apply softmax
    A = exp_max / np.sum(exp_max, axis=0, keepdims=True)
    return A, {"Z": Z}


def relu(Z: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # Apply relu
    A = np.maximum(0, Z)
    return A, {"Z": Z}


def forward_act(
        A_prev: np.ndarray, W: np.ndarray, B: np.ndarray,
        activation: str, shortcut: np.ndarray = None
) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    # Get activation function from activation string
    activation_func = softmax if activation == "softmax" else relu
    # Apply linear part of a layer's forward propagation
    Z, linear_cache = forward_lin(A_prev, W, B)
    # Apply activation part of a layer's forward propagation
    A, activation_cache = activation_func(Z)
    # Add shortcut connection if it exists
    if shortcut is not None:
        # Apply linear transformation to shortcut if dimensions don't match
        if shortcut.shape != A.shape:
            W_s = np.random.randn(A.shape[0], shortcut.shape[0])
            shortcut = np.dot(W_s, shortcut)
        A += shortcut
    return A, {'linear_cache': linear_cache,
               'activation_cache': activation_cache}


def forward_net_model(X, parameters, use_batchnorm=False, use_dropout=False):
    num_layers = len(parameters) // 2
    caches = []
    A = X
    for i in range(1, num_layers):
        A_prev = A
        A, cache = forward_act(A_prev, parameters[f"W{i}"], parameters[f"b{i}"], "relu", shortcut=A_prev)
        if use_batchnorm:
            A = apply_batchnorm(A)
        if use_dropout:
            D = np.random.rand(A.shape[0], A.shape[1]) < KEEP_PROB
            A = np.multiply(A, D)
            A /= KEEP_PROB
            cache["dropout"] = {"D": D}
        caches.append(cache)
    AL, cache = forward_act(A, parameters[f"W{num_layers}"], parameters[f"b{num_layers}"], "softmax")
    caches.append(cache)
    return AL, caches


def compute_cost(AL: np.ndarray, Y: np.ndarray) -> float:
    # Clipping Labels
    k = AL.shape[0]
    Y_clipped = np.clip(Y, (EPSILON / k), 1. - (EPSILON * ((k - 1) / k)))
    # Get number of examples
    m = AL.shape[1]
    # Calculate cost
    cost = -np.sum((Y_clipped * np.log(AL + EPSILON))) / m
    return cost



def apply_batchnorm(A: np.ndarray) -> np.ndarray:
    # Calculate mean, std, var
    sample_mean = A.mean(axis=0, keepdims=True)
    sample_var = A.var(axis=0, keepdims=True)
    std = np.sqrt(sample_var + EPSILON)

    # normalize
    A_centered = A - sample_mean
    A_norm = A_centered / std
    return A_norm


def backward_lin(dZ: np.ndarray,
                    cache: Tuple[np.ndarray, np.ndarray, np.ndarray]
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Unpack chache
    A_prev, W, b = cache
    # Get number of examples
    m = A_prev.shape[1]
    # Calculate gradient of the cost with respect to the activation
    dA_prev = np.dot(W.T, dZ)
    # Calculate gradient of the cost with respect to the W
    dW = (1.0 / m) * np.dot(dZ, A_prev.T)
    # Calculate gradient of the cost with respect to the b
    db = (1.0 / m) * np.sum(dZ, axis=-1, keepdims=True)
    return (dA_prev, dW, db)


def backward_act(
        dA: np.ndarray, cache: Dict[str, Dict[str, np.ndarray]],
        activation: str, shortcut: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Unpack linear_cache
    A_prev = cache['linear_cache']['A']
    W = cache['linear_cache']['W']
    b = cache['linear_cache']['b']
    # Get backward activation function from activation string
    activation_func_backward = (
        softmax_backward if activation == "softmax" else relu_backward)
    # Calculate gradient of the cost with respect to the linear
    # output of the current layer
    dZ = activation_func_backward(dA, cache['activation_cache'])
    # Add gradient from shortcut connection if it exists
    if shortcut is not None:
        dZ += shortcut
    return backward_lin(dZ, (A_prev, W, b))


def relu_backward(dA: np.ndarray, activation_cache: Dict[str, np.ndarray]
                  ) -> np.ndarray:
    # Get Z
    Z = activation_cache['Z']
    # Calculate gradient of the cost with respect to Z
    dZ = dA * (Z >= 0)
    return dZ


def tanh_backward(dA, cache):
    Z = cache["Z"]
    dtanh = 1 - np.square(np.tanh(Z))
    dZ = dA * dtanh
    return dZ





def softmax_backward(dA: np.ndarray, activation_cache: Dict[str, np.ndarray]
                     ) -> np.ndarray:
    # Get Z
    Z = activation_cache['Z']
    # Calculate gradient of the cost with respect to Z
    dZ = softmax(Z)[0] - dA
    return dZ


def backward_net_model(AL, Y, caches, use_dropout=False):
    grads = {}
    num_layers = len(caches)
    dA_prev, dW, db = backward_act(Y, caches[num_layers - 1], "softmax")
    grads[f"dA{num_layers}"] = dA_prev
    grads[f"dW{num_layers}"] = dW
    grads[f"db{num_layers}"] = db

    for i in reversed(range(num_layers - 1)):
        dA_prev = grads[f"dA{i + 2}"]
        if use_dropout:
            dA_prev = np.multiply(dA_prev, caches[i]["dropout"]["D"])
        # Pass the gradient of the cost with respect to the shortcut path
        dA_prev, dW, db = backward_act(dA_prev, caches[i], "relu", shortcut=dA_prev)
        grads[f"dA{i + 1}"] = dA_prev
        grads[f"dW{i + 1}"] = dW
        grads[f"db{i + 1}"] = db
    return grads

def param_update(parameters: Dict[str, np.ndarray],
                      grads: Dict[str, np.ndarray], learning_rate: float
                      ) -> Dict[str, np.ndarray]:
    # get number of layers
    num_layers = len(parameters) // 2

    # Loop over each layer
    for i in range(1, num_layers + 1):
        # Update W
        parameters[f"W{i}"] -= learning_rate * grads[f"dW{i}"]
        # Update b
        parameters[f"b{i}"] -= learning_rate * grads[f"db{i}"]
    return parameters


"""
# Example usage for normal neural network - Question 2.1
X = np.random.randn(10, 100)
Y = np.random.randn(10, 100)
layer_dims = [10, 5, 3, 10]

parameters = init_net(layer_dims)
AL, caches = forward_net_model(X, parameters, use_batchnorm=False, use_dropout=False)
grads_backprop = backward_net_model(AL, Y, caches, use_dropout=False)
updated_parameters = param_update(parameters, grads_backprop, learning_rate=0.01)



# Test with shortcuts - Resnet - Question 2.2
X = np.random.randn(10, 100)
Y = np.random.randn(10, 100)
layer_dims = [10, 5, 3, 10]
parameters = init_net(layer_dims)
AL, caches = forward_net_model(X, parameters, use_batchnorm=False, use_dropout=False)
grads_backprop = backward_net_model(AL, Y, caches, use_dropout=False)
updated_parameters = param_update(parameters, grads_backprop, learning_rate=0.01)
shortcut = X  # Initialize shortcut with the input X
for i in range(1, len(layer_dims)):
    A_prev = shortcut
    AL, cache = forward_act(A_prev, parameters[f"W{i}"], parameters[f"b{i}"], "relu", shortcut=shortcut)
    caches.append(cache)
    shortcut = AL  # Update shortcut with the output AL

# Backward pass with shortcuts
grads_backprop = backward_net_model(AL, Y, caches, use_dropout=False)

# Update parameters
updated_parameters = param_update(parameters, grads_backprop, learning_rate=0.01)
"""



# Jacobian Transpose test - Question 2.1 and 2.2
def f(x):
    return np.tanh(x)

def JacTMV(x, u):
    # The derivative of tanh(x) is 1 - tanh^2(x)
    return (1 - np.tanh(x) ** 2) * u

def g(x, u):
    return np.dot(f(x), u)

def g_grad(x, u):
    return JacTMV(x, u)

def jacobian_transposed_test(layer_activations, layer_gradients):
    epsilon = 0.1
    y0 = np.zeros(8)
    y1 = np.zeros(8)

    print("k\terror order 1\t\terror order 2")
    for k in range(1, 9):
        epsk = epsilon * (0.5 ** k)
        layer_activations_perturbed = layer_activations + epsk
        gk = layer_activations_perturbed
        g1 = layer_activations + epsk * layer_gradients
        y0[k - 1] = np.abs(gk - layer_activations).mean()  # Calculate the absolute difference element-wise and take the mean
        y1[k - 1] = np.abs(gk - g1).mean()  # Calculate the absolute difference element-wise and take the mean
        print(k, "\t", np.abs(gk - layer_activations).mean(), "\t", np.abs(gk - g1).mean())

    plt.semilogy(range(1, 9), y0, label="Zero order approx")
    plt.semilogy(range(1, 9), y1, label="First order approx")
    plt.legend()
    plt.title("Jacobian Transposed Test in semilogarithmic plot")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.show()






"""
# Test for 2.2.1 and 2.2.2
# Assuming you have computed the forward pass and gradients for each layer
layer_activations = AL
layer_gradients = grads_backprop["dA1"]

# Call the Jacobian Transpose test function for each layer
jacobian_transposed_test(layer_activations, layer_gradients)
"""


# Question 2.2.3:



def net_model(X: np.ndarray, Y: np.ndarray, layers_dims: List[int],
                  learning_rate: float, num_iterations: int, batch_size: int,
                  use_batchnorm: bool = False, use_dropout: bool = False,
                  validation_split: float = 0.2, shuffle: bool = True
                  ) -> Tuple[List[Tuple[float, float]], Dict[str, np.ndarray]]:
    # Initialize empty list for costs
    costs = []

    # Create validation and training set
    stack = np.hstack((X.T, Y.T))
    np.random.shuffle(stack)
    stack_size = int(np.floor(stack.shape[0] * validation_split))
    stack_train = stack[stack_size:]
    stack_val = stack[:stack_size]
    X_Val = stack_val[:, :X.shape[0]].T
    Y_Val = stack_val[:, X.shape[0]:].T

    # Initialize parameters
    parameters = init_net(layers_dims)
    # Keep track of number of batch
    batch_i = 0
    # track previous epoch val cost for early stopping
    prev_epoch_val_cost = np.Inf

    # loop over each epoch
    for epoch in range(1, num_iterations + 1):
        # Initialize list for storing all loss
        epoch_cost = []
        # Initialize list for storing all accuarcy
        epoch_acc = []
        # Shuffle X_train and Y_train together
        if shuffle:
            np.random.shuffle(stack_train)
        # Loop over batches
        for batch in split_given_size(stack_train, batch_size):
            # Get batch X and transpose it
            X_batch = batch[:, :X.shape[0]].T
            # Get batch Y and transpose it
            Y_batch = batch[:, X.shape[0]:].T
            # Forward propagation
            AL, caches = forward_net_model(X_batch, parameters,
                                         use_batchnorm, use_dropout)
            # Compute cost + track iteration cost
            cost = compute_cost(AL, Y_batch)
            epoch_cost.append(cost)
            if batch_i % 100 == 0:
                # Calculate Validation loss and accuracy
                np.random.shuffle(stack_val)
                batch_AL_val, _ = forward_net_model(
                    X_Val, parameters, use_batchnorm, False)
                batch_val_cost = compute_cost(batch_AL_val, Y_Val)
                costs.append((cost, batch_val_cost))
            batch_i += 1
            # compute accuracy + track iteration accuracy
            epoch_acc.append(calculate_accuracy(X_batch, Y_batch, parameters,
                                     use_batchnorm))
            # Backward propagation
            grads = backward_net_model(AL, Y_batch, caches, use_dropout)
            # Update parameters based on gradients
            parameters = param_update(parameters, grads, learning_rate)

        # Calculate Validation loss and accuracy
        AL_val, _ = forward_net_model(X_Val, parameters, use_batchnorm, False)
        epoch_val_cost = compute_cost(AL_val, Y_Val)
        epoch_val_acc = calculate_accuracy(X_Val, Y_Val, parameters, use_batchnorm)

        # Calculate Train loss and accurcy
        epoch_train_cost = np.mean(epoch_cost)
        epoch_train_acc = np.mean(epoch_acc)

        # Print epoch information
        print((f"Epoch (Training step) #{epoch} | train_loss: "
               f"{epoch_train_cost:.4f} - val_loss: {epoch_val_cost:.4f} | "
               f"train_acc: {(epoch_train_acc * 100):.4f}% - "
               f"val_acc: {(epoch_val_acc * 100):.4f}%"))

        # Check if we need to early stop
        if prev_epoch_val_cost - epoch_val_cost < EARLY_STOPPING_EPSILON:
            print("Early stopping!")
            print((f"Training has completed after {batch_i} iterations and "
                   f"{epoch} epochs (training steps)!"))
            return costs, parameters

        # Update
        prev_epoch_val_cost = epoch_val_cost
    print((f"Training has completed after {batch_i} iterations steps and "
           f"{num_iterations + 1} epochs (training steps)!"))
    return costs, parameters


def calculate_accuracy(X: np.ndarray, Y: np.ndarray,
            parameters: Dict[str, np.ndarray], use_batchnorm: bool = False
            ) -> float:
    # Forward propagation
    prob, caches = forward_net_model(X, parameters, use_batchnorm, False)
    # Get class from one hots for prediction
    pred_labels = np.argmax(prob, axis=0)
    # Get class from one hots for true labels
    true_labels = np.argmax(Y, axis=0)
    # calculate accuracy
    acc = (pred_labels == true_labels).sum() / pred_labels.shape[0]
    return acc


def split_given_size(arr: np.ndarray, size: int) -> List[np.ndarray]:
    return np.split(arr, np.arange(size, len(arr), size))


# Test for Question 2.2.3 and 2.2.4

"""
# Define model parameters
np.random.seed(0)
# This is for 2.2.4
#layer_dims = [X_train.shape[0], 2, 3, 4, 5] # Try 1
#layer_dims = [X_train.shape[0], 2, 2, 2,  3, 4, 5] # Try 2
#layer_dims = [X_train.shape[0], 3, 2, 2, 4, 2,  2, 5] # Try 3

# This is for 2.2.5
#layer_dims = [X_train.shape[0], 20, 7, 15, 5] # Try 1
#layer_dims = [X_train.shape[0], 20, 4, 7, 2, 3, 15, 5] # Try 2
#layer_dims = [X_train.shape[0], 10, 8, 2, 4, 7,  15, 5] # Try 3



learning_rate = 0.009
num_iterations = 100
batch_size = 32
batchnorm = True
dropout = True

# Train model
costs, parameters = net_model(X_train, C_train, layer_dims, learning_rate,
                                  num_iterations, batch_size,
                                  validation_split=0.2, shuffle=True,
                                  use_batchnorm=batchnorm,
                                  use_dropout=dropout)


# This is for 2.2.4
# plot cost vs training step
plt.figure(figsize=(20, 8))
plt.plot(list(range(0, len(costs) * 100, 100)),
         list(map(lambda x: x[0], costs)), c='blue', label='Train loss')
plt.plot(list(range(0, len(costs) * 100, 100)),
         list(map(lambda x: x[1], costs)),  c='orange', label='Validation loss')
plt.title('Train Loss vs Validation Loss Per Iteration', size=20)
plt.xlabel("Training Iteration")
plt.ylabel("Cost")
plt.legend()
plt.show()

# Calculate test accuracy
print(f"Test accuracy: {(calculate_accuracy(X_validation, C_validation, parameters, batchnorm) * 100):.4f}%")



# Test for question 2.2.3
def F(x): # Test for 2.2.3 - replaces the original F(x) for grad test in part 1
    return calculate_accuracy(X_validation, C_validation, parameters, batchnorm)

np.random.seed(0)
gradient_test(weights)


"""

