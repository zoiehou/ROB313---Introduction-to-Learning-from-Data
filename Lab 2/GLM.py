import numpy as np
import math
from matplotlib import pyplot as plt
from data_utils import load_dataset

#Function to compute the Root Mean Squared Error
def rmse(y_1, y_2):
    return np.sqrt(np.average((y_1-y_2)**2))


#Minkowski sum with p = 2
def minkowski_2(x_1, x_2):
    return np.linalg.norm([x_1-x_2], ord = 2)


#Customized Kernel
#Implement the Function: (1+x_1*x_2)^2 + x_1*x_2*cos(freq*(x_1-x_2))
def cus_k(x_1, x_2):
    freq = 0.0565
    k = (1+x_1*x_2)**2 + x_1*x_2*math.cos(2*math.pi/freq*(x_1-x_2))
    return k
    

#Vector consisting the basis functions
def basis_funcs(x):
    freq = 2*math.pi/0.0565 #Calculated frequency of mauna_loa oscillations
    row = list()
    row.append(1)
    #First helper basis function
    row.append((math.sqrt(2))*x)
    #Second
    row.append(x**2)
    #Third
    row.append(x*math.sin(freq*x))
    #Fourth
    row.append(x*math.cos(freq*x))
    vec = np.array(row)
    return vec

#Gaussian RBF Function
def g_rbf(x_1, x_2, theta):
    return math.exp(-((minkowski_2(x_1,x_2))**2)/theta)


#General Linear Model (GLM) Validation Function
def glm_valid(x_train, x_valid, y_train, y_valid, l_test = None):
    #Regularization Parameters
    if l_test == None:
        #Test 0 to 30 if not user-defined
        l_test = list(range(0,31))
    
    #Compute Phi Matrix
    phi_m = np.empty((len(x_train), 5)) #Empty
    for i in range(0, len(x_train)): #Fill in
        phi_m[i,:] = basis_funcs(x_train[i])
        
    #Compute Validation Phi Matrix
    phi_v = np.empty((len(x_valid), 5)) #Empty
    for i in range(0, len(x_valid)): #Fill in
        phi_v[i,:] = basis_funcs(x_valid[i])
        
    #Compute SVD
    U, S, V = np.linalg.svd(phi_m)
    
    #Compute Inverted Sigma
    sigma = np.diag(S)
    zeros = np.zeros([len(x_train)-len(S), len(S)])
    sigma = np.vstack([sigma, zeros])
        
    #Compute weights and predictions
    rmse_min = np.inf
    #With range of lambda values
    for l in l_test:
        temp_1 = np.dot(sigma.T, sigma)
        temp_2 = np.linalg.inv(temp_1 + l*np.eye(len(temp_1)))
        
        weight = np.dot(V.T, np.dot(temp_2, np.dot(sigma.T, np.dot(U.T, y_train))))
        pred = np.dot(phi_v, weight)
        
        rmse_cur = rmse(pred, y_valid)
        
        #Select the lambda value with the least RMSE
        if rmse_cur < rmse_min:
            rmse_min = rmse_cur
            l_min = l
    
    return l_min

#GLM Testing Function
def glm_test(x_train, x_valid, x_test, y_train, y_valid, y_test, l):
    x_set = np.vstack([x_train, x_valid])
    y_set = np.vstack([y_train, y_valid])
    
    #Compute Phi Matrix
    phi_m = np.empty((len(x_set), 5)) #Empty
    for i in range(0, len(x_set)): #Fill in
        phi_m[i,:] = basis_funcs(x_set[i])
    
    #Compute Test Phi Matrix
    phi_t = np.empty((len(x_test), 5)) #Empty
    for i in range(0, len(x_test)): #Fill in
        phi_t[i,:] = basis_funcs(x_test[i])
        
    #Compute SVD
    U, S, V = np.linalg.svd(phi_m)
    
    #Compute Inverted Sigma
    sigma = np.diag(S)
    zeros = np.zeros([len(x_set)-len(S), len(S)])
    sigma = np.vstack([sigma, zeros])
    
    #Compute Test Weights and Predictions
    temp_1 = np.dot(sigma.T, sigma)
    temp_2 = np.linalg.inv(temp_1 + l*np.eye(len(temp_1)))
        
    weight = np.dot(V.T, np.dot(temp_2, np.dot(sigma.T, np.dot(U.T, y_set))))
    pred = np.dot(phi_t, weight)
        
    error = rmse(pred, y_test) #Compute RMSE
    
    #Plot Results
    plt.figure(1)
    plt.plot(x_test, y_test, '-r', label = 'True Values')
    plt.plot(x_test, pred, '-b', label = 'Prediction Values')
    plt.title('Mauna_loa GLM Predictions with Lambda Value' + str(l))
    plt.xlabel('x_test')
    plt.ylabel('y')
    plt.legend(loc = 'upper right')
    plt.savefig('mauna_loa_glm_test.png')
    
    return error

#ernelized General Linear Model 
def glm_k(x_train, x_valid, x_test, y_train, y_valid, y_test, l):
    x_set = np.vstack([x_train, x_valid])
    y_set = np.vstack([y_train, y_valid])
    
    #Compute Gram Matrix
    G = np.empty((len(x_set), len(x_set)))
    prev = {} #Previously Computed Custom Kernels
    for i in range(0, len(x_set)):
        for j in range(0, len(x_set)):
            val_1 = x_set[i]
            val_2 = x_set[j]
            val_str_1 = str((val_1, val_2))
            val_str_2 = str((val_2, val_1))
            #Add to Previously Computed Dict if not already inside
            if val_str_1 not in prev: 
                prev[val_str_1] = cus_k(val_1, val_2)
                prev[val_str_2] = prev[val_str_1]
            #Add Computed Kernal to Gram Matrix
            G[i, j] = prev[val_str_1]
    
    #Cholesky Factorization of Gram Matrix + Lambda*Identity Matrix
    #Compute Lower Triangular
    R = np.linalg.cholesky((G + l*np.eye(len(G))))
    P = np.linalg.inv(R) #Inverse of Lower Triangular
    
    #Compute Estimation of Dual-Variables Alpha
    a = np.dot(np.dot(P.T, P), y_set)
    
    #Compute Predictions
    pred = np.empty(np.shape(y_test))
    for i in range(0, len(x_test)):
        #Compute Vector that consists the Kernel Products of x testing points
        #and x training points
        k_p = np.empty(np.shape(a))
        for j in range(0, len(x_set)):
            k_p[j] = cus_k(x_test[i], x_set[j])
        pred[i] = np.dot(k_p.T, a)
    
    #Compute the Test RMSE
    error = rmse(y_test, pred)
    
    #Plot Results
    plt.figure(2)
    plt.plot(x_test, y_test, '-r', label = 'True')
    plt.plot(x_test, pred, '-b', label = 'Prediction')
    plt.title('Mauna_Loa Kernelized GLM Predictions At Lambda ' + str(l))
    plt.xlabel('x_test')
    plt.ylabel('y')
    plt.legend(loc = 'upper right')
    plt.savefig('mauna_loa_glm_kernelized.png')
            
    return error    
            
        
#Visualize Kernel
def visual_k():
    for i in range(0,2):
        y_list = list()
        
        z = np.linspace(-0.1 + i, 0.1 + i, 100)
        z = np.array(z)
        
        for j in z:
            y_list.append(cus_k(i, j))
        #Plot    
        plt.figure(i+3)
        plt.plot(z, y_list, '-g', label='Kernel')
        plt.title('Kernel (' + str(i) + ', z+' + str(i) + ') Over z')
        plt.xlabel('z')
        plt.ylabel('k')
        plt.legend(loc = 'upper right')
        plt.savefig('kernel' + str(i) + '.png')


#Gaussion RBF GLM Validation Function
def g_rbf_glm_valid(x_train, x_valid, y_train, y_valid, data):
    theta = [0.05, 0.1, 0.5, 1, 2]
    reg = [0.001, 0.01, 0.1, 1]
    #Dict to store Validation Errors or Validation Accurary
    #for each Theta-Regularization Value
    result = {}
    
    #For each Theta Value
    for t in theta:
    #Compute Gram Matrix
        G = np.empty((len(x_train), len(x_train)))
        prev = {} #Previously Computed Gaussian Kernels
        for i in range(0, len(x_train)):
            for j in range(0, len(x_train)):
                val_1 = x_train[i]
                val_2 = x_train[j]
                val_str_1 = str((val_1, val_2))
                val_str_2 = str((val_2, val_1))
                #Add to Previously Computed Dict if not already inside
                if val_str_1 not in prev: 
                    prev[val_str_1] = g_rbf(val_1, val_2, t)
                    prev[val_str_2] = prev[val_str_1]
                #Add Computed Kernal to Gram Matrix
                G[i, j] = prev[val_str_1]
        
        #Compute k-vectors and its Matrix Form
        k_m = np.empty((len(x_valid), len(x_train)))
        for i in range(0, len(x_valid)):
            #Compute Kernel Vector
            #Kernel Products of x test points and x training points
            k_v = list()
            v = x_valid[i]
            for j in range(0, len(x_train)):
                k_v.append(g_rbf(v, x_train[j], t))
            k_m[i, :] = np.array(k_v)
        
        #For Current Theta
        #Compute Test Errors for Each Regularization Value
        for l in reg:
            #Cholesky Factorization of Gram Matrix + Lambda*Identity Matrix)
            #Compute Lower Triangle
            R = np.linalg.cholesky((G + l*np.eye(len(G))))
            P = np.linalg.inv(R) #Inverse of Lower Triangular
            #Compute Estimation of Dual-Variables Alpha
            a = np.dot(np.dot(P.T, P), y_train)
            
            #Compute Test RMSE (Regression Datasets)
            if data == 'mauna_loa' or data == 'rosenbrock':
                pred = np.dot(k_m, a) #Predictions
                #For Current Regularization-Theta Pair
                #Compute Validation Error (RMSE)
                result[(t, l)] = rmse(y_valid, pred)
            
            #Compute Test Accuracy Ratio (Classification Datasets)
            else:
                pred = np.argmax(np.dot(k_m, a), axis = 1) #Predictions
                y_v = np.argmax(1*y_valid, axis = 1)
                result[(t, l)] = (pred == y_v).sum()/len(y_v)
                
    #Compute Optimal Theta and Regularization Value
    #For Regression Datasets
    if data == 'mauna_loa' or data == 'rosenbrock':
        op = np.inf
        for t, l in result:
            #If the Theta and Regularization Value are smaller
            if result[(t, l)] < op:
                op = result[(t, l)] #Optimal Result
                op_t = t #Optimal Theta
                op_r = l #Optimal Regularization
    
    #For Classification Datasets
    else:
        op = np.NINF
        for t, l in result:
            if result[(t, l)] > op:
                op = result[(t, l)] #Optimal Result
                op_t = t #Optimal Theta
                op_r = l #Optimal Regularization
    
    return op_t, op_r, op
                

def g_rbf_glm_test(x_train, x_valid, x_test, y_train, y_valid, y_test, l, t, data):
    x_set = np.vstack([x_train, x_valid])
    y_set = np.vstack([y_train, y_valid])
    
    #Compute Gram Matrix
    G = np.empty((len(x_set), len(x_set)))
    prev = {} #Previously Computed Gaussian Kernels
    for i in range(len(x_set)):
        for j in range(len(x_set)):
            val_1 = x_set[i]
            val_2 = x_set[j]
            val_str_1 = str((val_1, val_2))
            val_str_2 = str((val_2, val_1))
            #Add to Previously Computed Dict if not already inside
            if val_str_1 not in prev: 
                prev[val_str_1] = g_rbf(val_1, val_2, t)
                prev[val_str_2] = prev[val_str_1]
            #Add Computed Kernal to Gram Matrix
            G[i, j] = prev[val_str_1]
    
    #Compute k-vectors and its Matrix Form
    k_m = np.empty((len(x_test), len(x_set)))
    for i in range(0, len(x_test)):
        #Compute Kernel Vector
        #Kernel Products of x test points and x training points
        k_v = list()
        v = x_test[i]
        for j in range(0, len(x_set)):
            k_v.append(g_rbf(v, x_set[j], t))
        k_m[i, :] = np.array(k_v)
        
    #Cholesky Factorization of Gram Matrix + Lambda*Identity Matrix
    #Compute Lower Triangular
    R = np.linalg.cholesky((G + l*np.eye(len(G))))
    P = np.linalg.inv(R) #Inverse of Lower Triangular
    
    #Compute Estimation of Dual-Variables Alpha
    a = np.dot(np.dot(P.T, P), y_set)
    
    #Compute Test RMSE for Regression Datasets
    if data == 'mauna_loa' or data == 'rosenbrock':
        pred = np.dot(k_m, a) #Predictions
        error = rmse(pred, y_test)
    
    #Compute Test Accurary Ratio for Classification Datasets
    else:
        pred = np.argmax(np.dot(k_m, a), axis = 1) #Predictions
        y_t = np.argmax(1*y_test, axis = 1)
        error = (pred == y_t).sum()/len(y_t)
    
    return error               

def greedy(x_train, x_valid, x_test, y_train, y_valid, y_test, shape):
    x_set = np.vstack([x_train, x_valid])
    y_set = np.vstack([y_train, y_valid])
    
    k = 0
    candid = basis_funcs(0) #Available Basis Functions
    select = [] #Selected Basis Function
    
    op_t, op_r, op = g_rbf_valid(x_train, x_valid, y_train, y_valid, 'rosenbrock', shape)
    err = g_rbf_test(x_train, x_valid, x_test, y_train, y_valid, y_test, op_r, shape, 'rosenbrock')
    e = (len(x_set)/2)*math.log10(op) + (k/2)*math.log10(len(x_set)) #Stopping Criterion
    
    #Keep looping if error is bigger than Stopping Criterion
    while err > e and k < 4:
        print(err, e)
        k = k+1
        select.append(candid[k])
        op_t, op_r, op = g_rbf_valid(x_train, x_valid, y_train, y_valid, 'rosenbrock', shape)
        err = g_rbf_test(x_train, x_valid, x_test, y_train, y_valid, y_test, op_r, shape, 'rosenbrock')
        e = (len(x_set)/2)*math.log10(op) + (k/2)*math.log10(len(x_set))
    
    return err,k


def g_rbf_valid(x_train, x_valid, y_train, y_valid, data, theta):
    reg = [0.001, 0.01, 0.1, 1]
    #Dict to store Validation Errors or Validation Accurary
    #for each Theta-Regularization Value
    result = {}
    
    #Compute Gram Matrix
    G = np.empty((len(x_train), len(x_train)))
    prev = {} #Previously Computed Gaussian Kernels
    
    for i in range(0, len(x_train)):
        for j in range(0, len(x_train)):
            val_1 = x_train[i]
            val_2 = x_train[j]
            val_str_1 = str((val_1, val_2))
            val_str_2 = str((val_2, val_1))
            #Add to Previously Computed Dict if not already inside
            if val_str_1 not in prev:
                prev[val_str_1] = g_rbf(val_1, val_2, theta)
                prev[val_str_2] = prev[val_str_1]
            #Add Computed Kernal to Gram Matrix
            G[i,j] = prev[val_str_1]
    
    #Compute k-vectors and its Matrix Form
    k_m = np.empty((len(x_valid), len(x_train)))
    for i in range(0, len(x_valid)):
        #Compute Kernel Vector
        #Kernel Products of x test points and x training points
        k_v = list()
        v = x_valid[i]
        for j in range(0, len(x_train)):
            k_v.append(g_rbf(v, x_train[j], theta))
        k_m[i,:] = np.array(k_v)
    
    #Compute Test Errors for Each Regularization Value
    for l in reg:
        #Cholesky Factorization of Gram Matrix + Lambda*Identity Matrix)
        #Compute Lower Triangle
        R = np.linalg.cholesky((G + l*np.eye(len(G))))
        P = np.linalg.inv(R) #Inverse of Lower Triangular
        a = np.dot(np.dot(P.T, P), y_train)
        
        #Compute Test RMSE
        pred = np.dot(k_m, a) #Predictions
        result[(theta, l)] = rmse(y_valid, pred)
    
    op = np.inf
    for theta, l in result:
        if result[(theta, l)] < op:
            #If the Theta and Regularization Value are smaller
            op = result[(theta, l)] #Optimal Result
            op_t = theta #Optimal Theta
            op_r = l #Optimal Regularization
            
    return op_t, op_r, op

def g_rbf_test(x_train, x_valid, x_test, y_train, y_valid, y_test, l, t, data):
    x_set = np.vstack([x_train, x_valid])
    y_set = np.vstack([y_train, y_valid])
    
    #Compute Gram Matrix
    G = np.empty((len(x_set), len(x_set)))
    prev = {} #Previously Computed Gaussian Kernels
    
    for i in range(len(x_set)):
        for j in range(len(x_set)):
            val_1 = x_set[i]
            val_2 = x_set[j]
            val_str_1 = str((val_1, val_2))
            val_str_2 = str((val_2, val_1))
            #Add to Previously Computed Dict if not already inside
            if val_str_1 not in prev:
                prev[val_str_1] = g_rbf(val_1, val_2, t)
                prev[val_str_2] = prev[val_str_1]
            #Add Computed Kernal to Gram Matrix
            G[i, j] = prev[val_str_1]
    
    #Compute k-vectors and its Matrix Form
    k_m = np.empty((len(x_test), len(x_set)))
    for i in range(0, len(x_test)):
        #Compute Kernel Vector
        #Kernel Products of x test points and x training points
        k_v = list()
        v = x_test[i]
        for j in range(0, len(x_set)):
            k_v.append(g_rbf(v, x_set[j], t))
        k_m[i, :] = np.array(k_v)
    
    #Cholesky Factorization of Gram Matrix + Lambda*Identity Matrix
    #Compute Lower Triangular
    R = np.linalg.cholesky((G + l*np.eye(len(G))))
    P = np.linalg.inv(R) #Inverse of Lower Triangular
    a = np.dot(np.dot(P.T, P), y_set)
    
    pred = np.dot(k_m, a)
    error = rmse(pred, y_test)
    
    return error


if __name__ == '__main__':
    #Datasets
    #All sets
    sets_tot = ['mauna_loa', 'rosenbrock', 'pumadyn32nm', 'iris', 'mnist_small']
    #Regression sets
    sets_reg = ['mauna_loa', 'rosenbrock', 'pumadyn32nm']
    #Classification sets
    sets_class = ['iris', 'mnist_small']
    
    """#Question 2 Results
    print('Question 2: ' + '\n')
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
    l = glm_valid(x_train, x_valid, y_train, y_valid)
    error_test = glm_test(x_train, x_valid, x_test, y_train, y_valid, y_test, l)
    print('Optimal Regularizarion Paramater Validation: ' + '\n' + str(l))
    print('Test RMSE: ' + '\n' + str(error_test) + '\n')"""
    
    """#Question 3 Results
    print('Question 3: ' + '\n')
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
    error_test = glm_k(x_train, x_valid, x_test, y_train, y_valid, y_test, 14)
    visual_k()
    print('Test RMSE: ' + '\n' + str(error_test) + '\n')"""
    
    """#Question 4 Results
    print('Question 4: ' + '\n')
    #Dataset mauna_loa
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
    theta, reg, error_valid = g_rbf_glm_valid(x_train, x_valid, y_train, y_valid, 'mauna_loa')
    error_test = g_rbf_glm_test(x_train, x_valid, x_test, y_train, y_valid, y_test, reg, theta, 'mauna_loa')
    print('Mauna_Loa Results: ' + '\n')
    print('Optimal Lengthscale: ' + '\n' + str(theta))
    print('Optimal Regularizar: ' + '\n' + str(reg))
    print('Validation RMSE: ' + '\n' + str(error_valid) + '\n')
    print('Test RMSE: ' + '\n' + str(error_test) + '\n' + '\n')

    #Dataset rosenbrock
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train = 1000, d = 2)
    theta, reg, error_valid = g_rbf_glm_valid(x_train, x_valid, y_train, y_valid, 'rosenbrock')
    error_test = g_rbf_glm_test(x_train, x_valid, x_test, y_train, y_valid, y_test, reg, theta, 'rosenbrock')
    print('Rosenbrock Results: ' + '\n')
    print('Optimal Lengthscale: ' + '\n' + str(theta))
    print('Optimal Regularizar: ' + '\n' + str(reg))
    print('Validation RMSE: ' + '\n' + str(error_valid) + '\n')
    print('Test RMSE: ' + '\n' + str(error_test) + '\n' + '\n')
    
    #Dataset iris
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    theta, reg, error_valid = g_rbf_glm_valid(x_train, x_valid, y_train, y_valid, 'iris')
    error_test = g_rbf_glm_test(x_train, x_valid, x_test, y_train, y_valid, y_test, reg, theta, 'iris')
    print('Iris Results: ' + '\n')
    print('Optimal Lengthscale: ' + '\n' + str(theta))
    print('Optimal Regularizar: ' + '\n' + str(reg))
    print('Validation Accuracy Ratio: ' + '\n' + str(error_valid) + '\n')
    print('Test Accuracy Ratio: ' + '\n' + str(error_test) + '\n' + '\n')"""
    
    """#Question 5 Results
    print('Question 5: ' + '\n')
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=200, d=2)
    #err_001, k = greedy(x_train, x_valid, x_test, y_train, y_valid, y_test, 0.01)
    #print(err_001, k)
    err_01, k = greedy(x_train, x_valid, x_test, y_train, y_valid, y_test, 0.1)
    print(err_01, k)
    err_10, k = greedy(x_train, x_valid, x_test, y_train, y_valid, y_test, 1.0)
    print(err_10, k)"""
