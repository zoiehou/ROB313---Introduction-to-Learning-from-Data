import autograd.numpy as np
from autograd import value_and_grad
from data_utils import load_dataset
from matplotlib import pyplot as plt
import time
import random
import math


#Sigmoid Function
def sigmoid(z):
    return 1/(1+np.exp(-z))


#Log-Likelidhood Function
def log_likelihood(est, actual):
    tot = 0
    for i in range(0, len(est)):
        tot += actual[i]*np.log(sigmoid(est[i]))+(1-actual[i])*np.log(1-sigmoid(est[i]))
    return tot


#Accuracy Ratio Function
def acc_ratio(y_est, y_test):
    return (y_est == y_test).sum()/len(y_test)


#Accuracy Ratio Function (Generalized)
def acc_ratio_2(w_1, w_2, w_3, b_1, b_2, b_3, x, y):
    f = np.exp(forward_pass(w_1, w_2, w_3, b_1, b_2, b_3, x))
    f = np.argmax(f, axis = 1)
    y = np.argmax(y, axis = 1)
    return (f == y).sum()/len(y)


#Update Weights Function
def up_weight(w, grad_w, rate, dir):
    return w - dir*rate*grad_w


#Function to compute the Root Mean Squared Error
def rmse(y_1, y_2):
    return np.sqrt(np.average((y_1-y_2)**2))


#Minkowski sum with p = 2
def minkowski_2(x_1, x_2):
    return np.linalg.norm([x_1-x_2], ord = 2)


#Logistic Regression Model Gradient Descent Function
def log_reg_gd(x_train, x_valid, x_test, y_train, y_valid, y_test, rates, model):
    #Variables Initialization
    time_list = []
    test_acc = []
    test_log = []
    neg_log = {}
    
    x_set = np.vstack([x_train, x_valid])
    y_set = np.vstack([y_train, y_valid])
    
    #Compute X matrix for all learning rates
    X = np.ones((len(x_train), len(x_train[0]) + 1))
    X[:, 1:] = x_train
    
    #Compute X matrix for test predictions
    X_test = np.ones((len(x_test), len(x_test[0]) + 1))
    X_test[:, 1:] = x_test
    
    for r in rates:
        w = np.zeros(np.shape(X[0, :]))
        neg_log[r] = []
        #Gradient Descent Process
        time_tot = 0
        tic = time.time()
        for iteration in range(0, 5000):
            #Linear Model Estimates for Training Set
            ests = np.dot(X, w)
            ests = ests.reshape(np.shape(y_train))
            #Stochastic Gradient Descent
            if model == 'SGD':
                #Compute Mini-batch size of 1
                i = random.randint(0, len(y_train) - 1)
                #Gradient of Log-Likelihood Function
                grad_log = (y_train[i] - sigmoid(ests[i]))*X[i, :]
            #Gradient Descent 
            elif model == 'GD':
                #Compute Full-batch
                grad_log = np.zeros(np.shape(w))
                for i in range(0, len(y_train)):
                    grad_log += (y_train[i] - sigmoid(ests[i]))*X[i, :]
            #Weights update
            w = np.add(w, r*grad_log)
            
            #Compute Total Used Time so far
            toc = time.time()
            time_tot += toc - tic
            
            #Compute Full-Batch Log-Likelihood
            Log = log_likelihood(ests, y_train)
            neg_log[r].append(-Log)
            
            #Resume Timer (for next iteration)
            tic = time.time()
          
        #Compute Final Time
        toc = time.time()
        time_tot += toc - tic
        time_list.append(time_tot) #List of all total times used
        
        #Test Accuracy Ratio
        test_est = np.dot(X_test, w)
        test_est = test_est.reshape(np.shape(y_test))
        pred = np.zeros(np.shape(y_test))
        for i in range(0, len(pred)):
            p = sigmoid(test_est[i])
            if p > 1/2:
                pred[i] = 1
            elif p < 1/2:
                pred[i] = 0
            else:
                pred[i] = -1
        
        #Compile total list of Test Accuracy and Test Log-Likelihood
        test_acc.append(acc_ratio(pred, y_test))
        test_log.append(log_likelihood(test_est, y_test))

    #Compute Final Accuracy Ratio, Log-Likelihood, Preferred Rates
    acc_ratio_final = max(test_acc)
    test_log_final = min(test_log)
    min_rates = []
    min_rates.append(rates[test_acc.index(acc_ratio_final)])
    min_rates.append(rates[test_log.index(test_log_final)])
    
    return neg_log, time_list, acc_ratio_final, min_rates, test_log_final


#Plotting Function for Logistic Regression Models
def log_reg_plt(loss, model):
    #X-axis
    x = list(range(5000))
    #Gradient Descent
    if model == 'GD':
        plt.figure(3)
        plt.title('GD Full-Batch Neg Log-Likelihood')
        plt.xlabel('Iterations')
        plt.ylabel('Neg Log-Likelihood')
        for r in loss:
            plt.plot(x, loss[r], label = 'Neg Log-L for n = ' + str(r))
        plt.legend(loc = 'upper right')
        plt.savefig('gd_log_likelihood.png')
    #Stochastic Gradient Descent
    if model == 'SGD':
        plt.figure(4)
        plt.title('SGD Full-Batch Neg Log-Likelihood')
        plt.xlabel('Iterations')
        plt.ylabel('Neg Log-Likelihood')
        for r in loss:
            plt.plot(x, loss[r], label = 'Neg Log-L for n = ' + str(r))
        plt.legend(loc = 'lower right')
        plt.savefig('sgd_log_likelihood.png')
    

def forward_pass(W1, W2, W3, b1, b2, b3, x):
    """
    forward-pass for an fully connected neural network with 2 hidden layers of M neurons
    Inputs:
        W1 : (M, 784) weights of first (hidden) layer
        W2 : (M, M) weights of second (hidden) layer
        W3 : (10, M) weights of third (output) layer
        b1 : (M, 1) biases of first (hidden) layer
        b2 : (M, 1) biases of second (hidden) layer
        b3 : (10, 1) biases of third (output) layer
        x : (N, 784) training inputs
    Outputs:
        Fhat : (N, 10) output of the neural network at training inputs
    """
    H1 = np.maximum(0, np.dot(x, W1.T) + b1.T) # layer 1 neurons with ReLU activation, shape (N, M)
    H2 = np.maximum(0, np.dot(H1, W2.T) + b2.T) # layer 2 neurons with ReLU activation, shape (N, M)
    Fhat = np.dot(H2, W3.T) + b3.T # layer 3 (output) neurons with linear activation, shape (N, 10)
    # #######
    # Note that the activation function at the output layer is linear!
    # Implimentation of a stable log-softmax activation function at the ouput layer
    # #######
    
    #Compute Maximum of Every Row and put in Matrix Form
    A = -1*np.ones(np.shape(Fhat))*np.max(Fhat, axis = 1)[:, np.newaxis]
    #Compute Matrix of Log-Sums
    log_sums = np.ones(np.shape(Fhat))*-1*np.log(np.sum(np.exp(np.add(Fhat, A)), axis=1))[:, np.newaxis]
    #Subtraction of Every Row (Element-Wise)
    #Numberically Stable
    Fhat = np.add(np.add(Fhat, A), log_sums)
    
    return Fhat


def negative_log_likelihood(W1, W2, W3, b1, b2, b3, x, y):
    """
    computes the negative log likelihood of the model `forward_pass`
    Inputs:
        W1, W2, W3, b1, b2, b3, x : same as `forward_pass`
        y : (N, 10) training responses
    Outputs:
        nll : negative log likelihood
    """
    Fhat = forward_pass(W1, W2, W3, b1, b2, b3, x)
    # ########
    # Note that this function assumes a Gaussian likelihood (with variance 1)
    # You must modify this function to consider a categorical (generalized Bernoulli) likelihood
    # ########
    #nll = 0.5*np.sum(np.square(Fhat - y)) + 0.5*y.size*np.log(2.*np.pi)(Gaussian likelihood)
    
    #Implementation of Categorical (Generalized Bernoulli) Likelihood
    #Compute Inner Product Vector
    inner_prod_v = np.einsum('ij, ij->i', Fhat, y)
    nll = np.sum(inner_prod_v)
    nnll = -1*nll
    return nnll
    

nll_gradients = value_and_grad(negative_log_likelihood, argnum=[0,1,2,3,4,5])
"""
    returns the output of `negative_log_likelihood` as well as the gradient of the 
    output with respect to all weights and biases
    Inputs:
        same as negative_log_likelihood (W1, W2, W3, b1, b2, b3, x, y)
    Outputs: (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad))
        nll : output of `negative_log_likelihood`
        W1_grad : (M, 784) gradient of the nll with respect to the weights of first (hidden) layer
        W2_grad : (M, M) gradient of the nll with respect to the weights of second (hidden) layer
        W3_grad : (10, M) gradient of the nll with respect to the weights of third (output) layer
        b1_grad : (M, 1) gradient of the nll with respect to the biases of first (hidden) layer
        b2_grad : (M, 1) gradient of the nll with respect to the biases of second (hidden) layer
        b3_grad : (10, 1) gradient of the nll with respect to the biases of third (output) layer
     """

    
def run_example():
    """
    This example demonstrates computation of the negative log likelihood (nll) as
    well as the gradient of the nll with respect to all weights and biases of the
    neural network. We will use 50 neurons per hidden layer and will initialize all 
    weights and biases to zero.
    """
    # load the MNIST_small dataset
    from data_utils import load_dataset
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')
    
    # initialize the weights and biases of the network
    M = 50 # 50 neurons per hidden layer
    W1 = np.zeros((M, 784)) # weights of first (hidden) layer
    W2 = np.zeros((M, M)) # weights of second (hidden) layer
    W3 = np.zeros((10, M)) # weights of third (output) layer
    b1 = np.zeros((M, 1)) # biases of first (hidden) layer
    b2 = np.zeros((M, 1)) # biases of second (hidden) layer
    b3 = np.zeros((10, 1)) # biases of third (output) layer
    
    # considering the first 250 points in the training set, 
    # compute the negative log likelihood and its gradients
    (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad)) = \
        nll_gradients(W1, W2, W3, b1, b2, b3, x_train[:250], y_train[:250])
    print("negative log likelihood: %.5f" % nll)

    
#Plot Generation Function    
def plots(i, loss, length, d_type, file):
    x_axis = list(range(length))
    if d_type == 'Train':
        norm = 10000
    if d_type == 'Valid':
        norm = 1000
        
    for rate in loss:
        loss[rate] = [x/norm for x in loss[rate]]
    
    #Plot
    plt.figure(i)
    plt.title('SGD (250) Full-Batch Normalized Neg Log-Likelihood')
    plt.xlabel('Iterations')
    plt.ylabel('Neg Log-Likelihood')
    for rate in loss:
        plt.plot(x_axis, loss[rate], label = (d_type + 'Neg LL n = ' + str(rate)))
    plt.legend(loc = 'upper right')
    plt.savefig(file + '.png')
                

#Plot Test Set Digits
def plot_digit(x, i, j, neuron = False):
    assert np.size(x) == 784
    plt.imshow(x.reshape((28, 28)), interpolation = 'none', aspect = 'equal', cmap = 'gray')
    
    if neuron:
        plt.savefig('w_1[neuron ' + str(i+1) + ' ].png')
    else:
        plt.savefig('test_digit_' + str(i) + '_rank_' + str(j) + '.png')


#Neural Network Function
def nn_sgd(x_train, x_valid, x_test, y_train, y_valid, y_test, M, B, l, rates, t=False, v=False, d=False):
    #Xavier Initialization for Weights
    #Zero Initialization for Biases
    w_1 = np.random.randn(M, 784)/np.sqrt(784)
    w_2 = np.random.randn(M, M)/np.sqrt(M)
    w_3 = np.random.randn(10, M)/np.sqrt(M)
    b_1 = np.zeros((M, 1))
    b_2 = np.zeros((M, 1))
    b_3 = np.zeros((10, 1))
    
    #Neg Log-Likelihood for Training and Validation Sets
    nll_t = {}
    nll_v = {}
    
    for r in rates:
        nll_t[r] = list()
        nll_v[r] = list()
        #Mimimum Validation Log-Likelihood
        ll_v_min = np.inf
        ll_v_it_min = 0
        for i in range(0, l):
            if not v:
                #Compute Full-Batch Neg Log-Likelihood for Training and Validation Sets
                nll_v_fb = negative_log_likelihood(w_1, w_2, w_3, b_1, b_2, b_3, x_valid, y_valid)
                nll_v[r].append(nll_v_fb)
                #Compute Minimum Log-Likelihood for Validation Set and Corresponding Weights
                if nll_v_fb < ll_v_min:
                    ll_v_min = nll_v_fb
                    ll_v_it_min = i
                    w_1_opt = w_1
                    w_2_opt = w_2
                    w_3_opt = w_3
                    b_1_opt = b_1
                    b_2_opt = b_2
                    b_3_opt = b_3
            #Compute a list of 250 random integers as the Mini-Batch Indices
            ind = np.random.choice(x_train.shape[0], size = B, replace = False)
            mini_b_x = x_train[ind, :]
            mini_b_y = y_train[ind, :]
            
            #Compute Log-Likelihood and Corresponding Gradients
            (nll, (w_1_g, w_2_g, w_3_g, b_1_g, b_2_g, b_3_g)) = nll_gradients(w_1, w_2, w_3, b_1, b_2, b_3, mini_b_x, mini_b_y)
            
            if not t and not v:
                nll_t[r].append(nll/250*10000)
            
            #Update Weights
            w_1 = up_weight(w_1, w_1_g, r, 1)
            w_2 = up_weight(w_2, w_2_g, r, 1)
            w_3 = up_weight(w_3, w_3_g, r, 1)
            b_1 = up_weight(b_1, b_1_g, r, 1)
            b_2 = up_weight(b_2, b_2_g, r, 1)
            b_3 = up_weight(b_3, b_3_g, r, 1)
        
        if not v and not d:
            #Print Results
            print('Results of ' + str(M) + ' Neuron w/ Learning Rate ' + str(r) + ':' + '\n')
            if not t:
                print('Train Neg Log-Likelihood: ' + str(nll_t[r][-1]) + '\n')
            print('Valid Neg Log-Likelihood: ' + str(nll_v[r][-1]) + '\n')
            print('Minimum Valid Neg Log-Likelihood: ' + str(ll_v_min) + ' at Iteration ' + str(ll_v_it_min + 1) + '\n')
            
            #Compute Optimal Validation and Test Sets Log-Likelihood and Accuracy Ratio
            if t:
                ratio_v = acc_ratio_2(w_1_opt, w_2_opt, w_3_opt, b_1_opt, b_2_opt, b_3_opt, x_valid, y_valid)
                ratio_test = acc_ratio_2(w_1_opt, w_2_opt, w_3_opt, b_1_opt, b_2_opt, b_3_opt, x_test, y_test)
                nll_test = negative_log_likelihood(w_1_opt, w_2_opt, w_3_opt, b_1_opt, b_2_opt, b_3_opt, x_test, y_test)
                print('Optimal Validation Ratio: ' + str(ratio_v) + '\n')
                print('Optimal Test Ratio: ' + str(ratio_test) + '\n')
                print('Optimal Test Neg Log-Likelihood: ' + str(nll_test) + ' at Iteration ' + str(ll_v_it_min + 1))
                
        if d:
            F = np.max(np.exp(forward_pass(w_1_opt, w_2_opt, w_3_opt, b_1_opt, b_2_opt, b_3_opt, x_test)), axis = 1)
            ind_sorted = np.argsort(F)
            test_sorted = x_test[ind_sorted]
        print('\n')
    
    if d:
        return ind_sorted, test_sorted
    
    if not v:
        if not t:
            return nll_t, nll_v
        else:
            return nll_v
    else:
        seen = list()
        for i in range(0, 17):
            j = np.random.randint(M)
            if j not in seen:
                seen.append(j)
                plot_digit(w_1[j], j, 0, neuron = True)
            else:
                i -= 1



if __name__ == '__main__':
    #Datasets
    #All sets
    sets_tot = ['mauna_loa', 'rosenbrock', 'pumadyn32nm', 'iris', 'mnist_small']
    #Regression sets
    sets_reg = ['mauna_loa', 'rosenbrock', 'pumadyn32nm']
    #Classification sets
    sets_class = ['iris', 'mnist_small']
    
    #Question 1 Results
    print('Question 1: ' + '\n')
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    y_train, y_valid, y_test = y_train[:, (1,)], y_valid[:, (1,)], y_test[:, (1,)]
    y_train = np.asarray(y_train, int)
    y_valid = np.asarray(y_valid, int)
    y_test = np.asarray(y_test, int)
    rates = [0.0001, 0.001, 0.01]
    
    neg_log_gd, time_gd, ratio_gd, min_rates_gd, test_log_gd = log_reg_gd(x_train, x_valid, x_test, y_train, y_valid, y_test, rates, 'GD')
    #neg_log_sgd, time_sgd, ratio_sgd, min_rates_sgd, test_log_sgd = log_reg_gd(x_train, x_valid, x_test, y_train, y_valid, y_test, rates, 'SGD')

    log_reg_plt(neg_log_gd, 'GD')
    #log_reg_plt(neg_log_sgd, 'SGD')
    
    print('Full-Batch Gradient Descent Results: ' + '\n')
    print('Test Accuracy Ratio: ' + '\n' + str(ratio_gd) + '\n')
    print('Test Log-likelihood: ' + '\n' + str(test_log_gd) + '\n')
    print('Convergence Times: ' + '\n' + str(time_gd) + '\n')
    print('Preferred Rate (Gradient Descent - Ratio): ' + '\n' + str(min_rates_gd[0]) + '\n')
    print('Preferred Rate (Gradient Descent - Log): ' + '\n' + str(min_rates_gd[1]) + '\n')
    
    """print('Stochastic Gradient Descent Results: ' + '\n')
    print('Test Accuracy Ratio: ' + '\n' + str(ratio_sgd) + '\n')
    print('Test Log-likelihood: ' + '\n' + str(test_log_sgd) + '\n')
    print('Convergence Times: ' + '\n' + str(time_sgd) + '\n')
    print('Preferred Rate (Gradient Descent - Ratio): ' + '\n' + str(min_rates_sgd[0]) + '\n')
    print('Preferred Rate (Gradient Descent - Log): ' + '\n' + str(min_rates_sgd[1]) + '\n')"""
    

    """#Question 2 Results
    print('Question 2: ' + '\n')
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')
    nll_t, nll_v = nn_sgd(x_train, x_valid, x_test, y_train, y_valid, y_test, 100, 250, 4000, [0.0001])
    plots(1, nll_t, 4000, 'Train', 'mini_b_250_nn_100_0.0001')
    plots(1, nll_v, 4000, 'Valid', 'mini_b_250_nn_100_0.0001')
    nll_t, nll_v = nn_sgd(x_train, x_valid, x_test, y_train, y_valid, y_test, 100, 250, 1000, [0.001])
    plots(1, nll_t, 1000, 'Train', 'mini_b_250_nn_100_0.001')
    plots(1, nll_v, 1000, 'Valid', 'mini_b_250_nn_100_0.001')
    
    ind, test_sorted = nn_sgd(x_train, x_valid, x_test, y_train, y_valid, y_test, 100, 250, 1000, [0.001], t = True, v = False, d = True)
    for i in range(0, 20):
        plot_digit(test_sorted[i], ind[i], i)"""
