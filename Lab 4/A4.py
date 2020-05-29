import numpy as np
from data_utils import load_dataset
from matplotlib import pyplot as plt
import math
import scipy


#Sigmoid Function
def sigmoid(z):
    return np.divide(1, np.add(1, np.exp(-1*z)))


#Log-Likelidhood Function
def log_likelihood(est, act):
    tot_1 = np.dot(act.T, np.log(sigmoid(est)))
    tot_2 = np.dot(np.subtract(1, act).T, np.log(np.subtract(1, sigmoid(est))))
    tot = np.add(tot_1, tot_2)
    return tot


#Log-Likelihood Gradient
def likelihood_grad(X, est, act):
    grad = np.zeros(np.shape(X[0]))
    for i in range(0, len(est)):
        grad += (act[i]-sigmoid(est[i]))*X[i]
    return grad


#Log-Likelihood Hessian
def likelihood_hess(X, est):
    hess = np.zeros((len(X[0]), len(X[0])))
    v = np.multiply(sigmoid(est), np.subtract(sigmoid(est), 1))
    for i in range(0, len(est)):
        hess = np.add(hess, v[i]*np.outer(X[i], X[i].T))
    return hess


#Log-Likelihood Prior
def log_prior(w, sig):
    D_1 = len(w)
    tot_1 = -((D_1)/2)*np.log(2*math.pi)
    tot_2 = (D_1)/2*np.log(sig)
    tot_3 = np.divide(np.dot(w.T, w), 2*sig)
    tot = tot_1 - tot_2 - tot_3
    return tot


#Log-Likelihood Prior Gradient
def prior_grad(w, sig):
    return -np.divide(w, sig)


#Log-Likelihood Prior Hessian
def prior_hess(w, sig):
    return np.multiply(np.divide(-1, sig), np.eye(len(w)))


#Likelihood Prior
def prior_like(w, var):
    prior = 1
    for i in range(0, len(w)):
        prior *= 1/math.sqrt(2*math.pi*var)*math.exp(-(w[i]**2)/(2*var))
    return prior


#Matrix Form
def X_mat(data):
    X = np.ones((len(data), len(data[0])+1))
    X[:, 1:] = data
    return X


#Log Marginal Likelihood Helper
def log_g(hess):
    return -1/2*np.log(np.linalg.det(-1*hess))+len(hess)/2*np.log(2*np.pi)


#Proposal Distribution
def prop(mean, variance):
    return np.random.multivariate_normal(mean=mean, cov=np.eye(np.shape(mean)[0])*variance)


#Proposal Distribution Likelihood
def prop_like(w, prop_var, mean):
    prop = 1
    for i in range(0, len(w)):
        prop *= 1/math.sqrt(2*math.pi*prop_var)*math.exp(-((mean[i]-w[i])**2)/(2*prop_var))
    return prop


#Sample Weights
def sample_w(size, mean, variance):
    w = []
    for i in range(0, size):
        w.append(prop(mean, variance))
    return w

#Bernoulli Likelihood
def likelihood(x,y):
    ll = 1
    for i in range(len(x)):
        ll *= (sigmoid(x[i])**y[i])*((1-sigmoid(x[i]))**(1-y[i]))
    return ll


#r in Monte Carlo Sampler
def r(x, y, w, prior_var, prop_var, mean):
    return likelihood(x, y)*prior_like(w, prior_var)/prop_like(w, prop_var, mean)

#Compute Likelihood
def ll(y_pred, y):
    return np.dot(y.T, np.log(y_pred)) + np.dot(np.subtract(1, y).T, np.log(np.subtract(1, y_pred)))


#Accuracy Ratio Function
def acc_ratio(y_est, y_test):
    return (y_est == y_test).sum()/len(y_test)


#Laplace Approximation Function
def laplace_approx(x_train, x_test, y_train, y_test, rate):
    var_s = [0.5, 1, 2] #Prior Variances
    #Matrix Form
    X_train = X_mat(x_train)
    X_test = X_mat(x_test)
    marg = {} #Marginal Likelihoods
    it_list = {}
    for var in var_s:
        #Variables Initialization
        w = np.zeros(np.shape(X_train[0])) #Weights
        it = 0 #Number of Iteration
        #Compute First Gradient
        x = np.reshape(np.dot(X_train, w), np.shape(y_train)) #First Estimate
        post = likelihood_grad(X_train, x, y_train) + prior_grad(w, var)
        while max(post) > 10**(-9):
            #Break when gradients are nearly zero
            x = np.dot(X_train, w)
            post = likelihood_grad(X_train, x, y_train) + prior_grad(w, var)
            w = np.add(w, rate*post) #Update Weight
            it += 1
        #Compute Hessian at MAP Solution
        hess = likelihood_hess(X_train, x) + prior_hess(w, var)
        #Compute Marginal Likelihood\
        marg[var] = log_likelihood(x, y_train) + log_prior(w, var) + log_g(hess)
        it_list[var] = it
        
    return marg, it_list


#Importance Sampling Function
def import_samp(x_train, x_valid, x_test, y_train, y_valid, y_test, mean, sam_range=[5, 10, 50, 100, 500], v=False):
    var_prior = 1 #Prior Variance
    var_s = [1, 2, 5] #Proposal Distribution Metrics
    min_ll = np.inf #Minimum Log-Likelihood
    #Y Sets and X sets
    y_train = np.asarray(y_train, int)
    y_valid = np.asarray(y_valid, int)
    y_test = np.asarray(y_test, int)
    X_train = X_mat(x_train)
    X_valid = X_mat(x_valid)
    X_test = X_mat(x_test)
    

    for var in var_s:
        for s in sam_range:
            #Test Set Classification Storage
            pred = np.zeros(np.shape(y_valid))
            pred_discrete = np.zeros(np.shape(y_valid))
            #Sample Weights
            w = sample_w(s, mean, var)
            #Compute Predictions
            for d in range(0, len(X_valid)):
                #Inner Sample
                r_sum = 0
                for i in range(0, s):
                    r_sum += r(np.dot(X_train, w[i]), y_train, w[i], var_prior, var, mean)
                #Outer Sample    
                pred_sum = 0
                for j in range(0, s):
                    #Sigmoid Prediction of Test Point
                    y_star = sigmoid(np.dot(X_valid[d], w[j]))
                    pred_sum += y_star*r(np.dot(X_train, w[j]), y_train, w[j], var_prior, var, mean)/r_sum
                    
                #Classifications
                pred[d] = pred_sum
                if pred_sum > 0.5:
                    pred_discrete[d] = 1
                elif pred_sum < 0.5:
                    pred_discrete[d] = 0
                else:
                    pred_discrete[d] = -1
            #Valid Negative Log-Likelihood
            ll_valid = -ll(pred, y_valid)
            if ll_valid < min_ll:
                min_ll = ll_valid
                min_acc = acc_ratio(pred_discrete, y_valid)
                opt_var = var
                opt_size = s
        
    #Training and Validation Sets Merged
    x_train = np.vstack((x_train, x_valid))
    X_train = X_mat(x_train)
    y_train = np.vstack((y_train, y_valid))
    #Test Set Classification Storage
    pred_test = np.zeros(np.shape(y_test))
    pred_d_test = np.zeros(np.shape(y_test))
    #Sample Weights
    w = sample_w(opt_size, mean, opt_var)
    #Compute Predictions
    for d in range(0, len(X_test)):
        #Inner Sample
        r_sum = 0
        for i in range(0, opt_size):
            r_sum += r(np.dot(X_train, w[i]), y_train, w[i], var_prior, opt_var, mean)
        #Outer Sample    
        pred_sum = 0
        for j in range(0, opt_size):
            #Sigmoid Prediction of Test Point
            y_star = sigmoid(np.dot(X_test[d], w[j]))
            pred_sum += y_star*r(np.dot(X_train, w[j]), y_train, w[j], var_prior, opt_var, mean)/r_sum
                    
        #Classifications
        predict = pred_sum
        pred_test[d] = predict
        if predict > 0.5:
            pred_d_test[d] = 1
        elif predict < 0.5:
            pred_d_test[d] = 0
        else:
            pred_d_test[d] = -1
    #Test Negative Log-Likelihood    
    ll_test = -ll(pred_test, y_test)
    acc_test = acc_ratio(pred_d_test, y_test)

    #If Visualizations are needed
    if v:
        #Sample Weights
        w = sample_w(5000, mean, 2)
        r_sum = 0
        #Inner Sample
        for i in range(0, 5000):
            r_sum += r(np.dot(X_train, w[i]), y_train, w[i], var_prior, 2, mean)
        #Outer Sample
        post = []
        for j in range(0, 5000):
            post.append(r(np.dot(X_train, w[j]), y_train, w[j], var_prior, 2, mean)/r_sum)
        
        visual(mean, 2, post, w)
    
    return ll_test, acc_test, opt_var, opt_size, min_ll, min_acc


#Helper Visualization Function
def visual(mean, var, post, w):
    for i in range(0, 5):
        w_s = []
        for j in range(0, len(w)):
            w_s.append(w[j][i])
        #Zip Weights and Posterior
        w_s, post = zip(*sorted(zip(w_s, post)))
        #Gaussian Set-Up
        z = np.polyfit(w_s, post, 1)
        z = np.squeeze(z)
        p = np.poly1d(z)
        #Plot
        w_tot = np.arange(min(w_s), max(w_s), 0.001)
        q_w = scipy.stats.norm.pdf(w_tot, mean[i], var)
        plt.figure(i)
        plt.title('Posterior Visualization: q(w) mean = ' + str(round(mean[i], 2)) + ' var = ' + str(var))
        plt.xlabel('w[' + str(i+1) + ']')
        plt.ylabel('Probability')
        plt.plot(w_tot, q_w, '-y', label = 'Proposal q(w)')
        plt.plot(w_s, post, 'ob', label = 'Posterior P(w|X,y)')
        plt.plot(w_s, p(w_s), 'r--')
        plt.legend(loc = 'upper right')
        plt.savefig('weight_visual_' + str(i+1) + '.png')
        
        

if __name__ == '__main__':
    #Datasets
    #All sets
    sets_tot = ['mauna_loa', 'rosenbrock', 'pumadyn32nm', 'iris', 'mnist_small']
    #Regression sets
    sets_reg = ['mauna_loa', 'rosenbrock', 'pumadyn32nm']
    #Classification sets
    sets_class = ['iris', 'mnist_small']
    
    #Question 1(a)
    Q1_a = True
    #Question 1(b)
    Q1_b = False
    #Question 1(b) Visualization
    Q1_b_visual = False
    
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    y_train, y_valid, y_test = y_train[:, (1,)], y_valid[:, (1,)], y_test[:, (1,)]
    
    #Question 1(a) Results
    if Q1_a:
        print('Question 1(a): ' + '\n')
        x_train, x_test = np.vstack((x_train, x_valid)), x_test
        y_train, y_test = np.vstack((y_train, y_valid)), y_test
        marginal_likelihood, it_list = laplace_approx(x_train, x_test, y_train, y_test, 0.001)
        for var in marginal_likelihood:
            print('For Variance: ' + str(var) + ', w/ a Marginal Likelihood of ' + str(marginal_likelihood[var]) + ', Iteration: ' + str(it_list[var]) + '\n')

    #Question 1(b) Results
    if Q1_b:
        print('Question 1(b): ' + '\n')
        mean_MAP = [-0.87805271, 0.29302957, -1.2347739, 0.67815586, -0.89401743]
        if Q1_b_visual:
            ll_test, acc_test, opt_var, opt_size, min_ll, min_acc = import_samp(x_train, x_valid, x_test, y_train, y_valid, y_test, mean_MAP, v=True)
        else:
            ll_test, acc_test, opt_var, opt_size, min_ll, min_acc = import_samp(x_train, x_valid, x_test, y_train, y_valid, y_test, mean_MAP)
    
        print('Proposal Distribution Variance: ' + str(opt_var))
        print('Sample Size: ' + str(opt_size))
        print('Validation Log-Likelihood: ' + str(min_ll))
        print('Validation Accuracy Ratio: ' + str(min_acc))
        print('Test Log-Likelihood: ' + str(ll_test))
        print('Test Accuracy Ratio: ' + str(acc_test) + '\n')
