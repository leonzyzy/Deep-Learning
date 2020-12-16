# packages
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from jedi.api.refactoring import inline

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def update_parameters_with_gd(params, grads, eta):
    # get the number of total layers in the neural networks
    L = len(params) // 2

    # update rule for each parameter
    for l in range(L):
        params['W' + str(l + 1)] = params['W' + str(l + 1)] - eta * grads['dW' + str(l + 1)]
        params['b' + str(l + 1)] = params['b' + str(l + 1)] - eta * grads['db' + str(l + 1)]
    return params


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    mini_batches = []
    m = X.shape[1]  # number of instances

    # step1: Shuffle(X,Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # step2: Partition (shuffled_X,shuffled_Y)
    num_complete_minibatches = m // mini_batch_size  # number of mini-batches
    for k in range(num_complete_minibatches):
        # [0,1*64),[1*64:2*64),[2*64:3*64)...
        mini_batch_X = shuffled_X[:, k * (mini_batch_size):(k + 1) * (mini_batch_size)]
        mini_batch_Y = shuffled_Y[:, k * (mini_batch_size):(k + 1) * (mini_batch_size)]
        # put them into a tuple and push into list
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # step3: check end case
    # for last mini-batch < mini_batch_size, it should be [N*mini_batch_size,end]
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def initialize_velocity(params):
    L = len(params)//2
    # initialize all layers
    v = {}
    for l in range(L):
        v['dW'+str(l+1)] = np.zeros(params['W'+str(l+1)].shape)
        v['db'+str(l+1)] = np.zeros(params['b'+str(l+1)].shape)
    return v

def update_parameters_with_momentum(params,grads,v,beta,eta):
    L = len(params)//2
    for l in range(L):
        # updating v
        v['dW'+str(l+1)] = beta*v['dW'+str(l+1)]+(1-beta)*grads['dW'+str(l+1)]
        v['db'+str(l+1)] = beta*v['db'+str(l+1)]+(1-beta)*grads['db'+str(l+1)]
        # updating parameters
        params['W'+str(l+1)] = params['W'+str(l+1)]-eta*v['dW'+str(l+1)]
        params['b'+str(l+1)] = params['b'+str(l+1)]-eta*v['db'+str(l+1)]
    return params,v

def initialize_adam(params):
    L = len(params)//2
    v = {} # direction
    s = {} # variance
    for l in range(L):
        v['dW'+str(l+1)] = np.zeros(params['W'+str(l+1)].shape)
        v['db'+str(l+1)] = np.zeros(params['b'+str(l+1)].shape)
        s['dW'+str(l+1)] = np.zeros(params['W'+str(l+1)].shape)
        s['db'+str(l+1)] = np.zeros(params['b'+str(l+1)].shape)
    return v,s


def update_parameters_with_adam(params, grads, v, s, t, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(params) // 2
    v_hat = {}
    s_hat = {}
    for l in range(L):
        # updating v, the direction side
        v['dW' + str(l + 1)] = beta1 * v['dW' + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v['db' + str(l + 1)] = beta1 * v['db' + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        # updating s, the variance side
        s['dW' + str(l + 1)] = beta2 * s['dW' + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
        s['db' + str(l + 1)] = beta2 * s['db' + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)

        # bias correction
        v_hat['dW' + str(l + 1)] = v['dW' + str(l + 1)] / (1 - pow(beta1, t))
        v_hat['db' + str(l + 1)] = v['db' + str(l + 1)] / (1 - pow(beta1, t))
        s_hat['dW' + str(l + 1)] = s['dW' + str(l + 1)] / (1 - pow(beta2, t))
        s_hat['db' + str(l + 1)] = s['db' + str(l + 1)] / (1 - pow(beta2, t))

        # updating parameters
        deltaW = np.divide(v_hat['dW' + str(l + 1)], np.sqrt(s_hat['dW' + str(l + 1)]) + epsilon)
        deltab = np.divide(v_hat['db' + str(l + 1)], np.sqrt(s_hat['db' + str(l + 1)]) + epsilon)
        params['W' + str(l + 1)] = params['W' + str(l + 1)] - eta * deltaW
        params['b' + str(l + 1)] = params['b' + str(l + 1)] - eta * deltab
    return params, v, s


# get training data
train_X, train_Y = load_dataset()


def model(X, Y, layers_dim, optimizer, eta=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    L = len(layers_dim)  # number of layers
    costs = []  # cost function
    t = 0  # initial t
    seed = 10

    # get parameters
    params = initialize_parameters(layers_dim)

    # define optimizer
    if optimizer == 'gd':  # gradient descent
        pass  # no initialization required for gradient descent
    elif optimizer == 'momentum':
        v = initialize_velocity(params)
    elif optimizer == 'adam':
        v, s = initialize_adam(params)

    # training neural net work
    for i in range(num_epochs):
        # We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        # setup mini-bacthes training
        minibacthes = random_mini_batches(X, Y, mini_batch_size, seed)
        for minibatch in minibacthes:
            # get X and Y in each mini-bacth
            (minibatch_X, minibatch_Y) = minibatch
            # forward propagation
            a3, caches = forward_propagation(minibatch_X, params)
            # compute cost
            cost = compute_cost(a3, minibatch_Y)
            # backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)
            # updating parameters
            if optimizer == 'gd':  # gradient descent
                params = update_parameters_with_gd(params, grads, eta)
            elif optimizer == 'momentum':
                params = update_parameters_with_momentum(params, grads, v, beta, eta)
            elif optimizer == 'adam':
                params = update_parameters_with_adam(params, grads, v, s, t, eta, beta1, beta2, epsilon)
        # print cost
        if print_cost and i % 1000 == True:
            print("Cost after epoch %i: %f" % (i, cost))
        # append cost
        if print_cost and i % 100 == 0:
            costs.append(cost)
        # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(eta))
    plt.show()
    return params
