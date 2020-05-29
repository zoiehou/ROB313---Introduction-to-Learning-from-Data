import numpy as np
import math
import time
from matplotlib import pyplot as plt
from data_utils import load_dataset
from sklearn import neighbors #this is only allowed for question 3

#3 distance metrics
#Minkowski sum with p = 1
def minkowski_1(x1,x2):
    return np.linalg.norm([x1-x2], 1)

#Minkowski sum with p = 2
def minkowski_2(x1,x2):
    return np.linalg.norm([x1-x2], 2)

#Function to compute the Root Mean Squared Error
def rmse(y1,y2):
    return np.sqrt(np.average((y1-y2)**2))

#Five-Fold Cross-Validation Function
def ff_cross_valid(x_train, x_valid, y_train, y_valid, dist_met, k_list = None):
    #Variables Initialization
    #A dict for Root Mean Squared Error values
    rmse_vals = {}
    err_final = []
    
    #Check if user inputs a k_list
    #If not, iterate through k = 0 to 25 to check for the best k
    #The range of k's can be changed
    if k_list == None:
        k_list = list(range(0,26))
    
    #Merge train and valid sets together
    x_set = np.vstack([x_train, x_valid])
    y_set = np.vstack([y_train, y_valid])
    
    #Shuffle datasets with seed
    np.random.seed(7)
    np.random.shuffle(x_set)
    np.random.seed(7)
    np.random.shuffle(y_set)
        
    #Divide the set into 5 groups for k-Fold Cross-Validation (k = 5)
    set_length = len(x_set)//5
    
    #Iterate over the 5 folds
    for i in range(0,5):
        #Compute train and valid sets with each fold
        #x sets
        x_valid = x_set[i*set_length : (i+1)*set_length]
        x_train = np.vstack([x_set[:i*set_length], x_set[(i+1)*set_length:]])
        #y sets
        y_valid = y_set[i*set_length : (i+1)*set_length]
        y_train = np.vstack([y_set[:i*set_length], y_set[(i+1)*set_length:]])

        #Computation of distances between points in train and valid sets using distance metrics
        for met in dist_met:
            estim = {}
            #Compute distances between training x set points and each validation point
            for j in range(0,set_length):
                x_dist_list = []
                for d in range(0,len(x_train)):
                    x_dist = met(x_train[d], x_valid[j])
                    x_dist_list.append((x_dist, y_train[d]))
                
                #Sort the x distance list from smallest to largest distance
                x_dist_list.sort(key=lambda x: x[0])
                
                #Compute k-NN
                for k in k_list:
                    new_dist_list = x_dist_list[:k+1]
                    y_tot = []
                    #Check if the dict already has a value with key k
                    #If not, compute one
                    if k not in estim: 
                        estim[k] = []
                    #Add in all y values to the y_tot list
                    for y_dist in new_dist_list:
                        y_tot.append(y_dist[1])
                    #Compute the average for the y values
                    y_avg = sum(y_tot)/len(y_tot)
                    #Add the average value for y into the dict with the key k
                    estim[k].append(y_avg)
                    
            #Compute the Root Mean Squared Error for each k
            for k in k_list:
                #Check if the dict already has a RMSE value with the key (dist_met, k)
                #If not, compute one
                if (met, k) not in rmse_vals:
                    rmse_vals[(met,k)] = []
                #Compute the RMSE value
                rmse_val = rmse(y_valid, estim[k])
                #Add the RMSE value into the dict containing RMSE values for all k's with key (dist_met, k)
                rmse_vals[(met, k)].append(rmse_val)
                        
    #Compute average error list
    for met, k in rmse_vals:
        #Compute the average for the RMSE values across 5 folds
        error_avg = sum(rmse_vals[met, k])/len(rmse_vals[met, k])
        #Add the average error value into the final error list
        err_final.append((k+1, met, error_avg))
    
    return err_final

#Testing Regression Function
def regression_test(x_train, x_valid, x_test, y_train, y_valid, y_test, k, dist_met, plot = False):
    #Variables Initialization
    #Prediction list to store the prediction for each data point
    pred = []    
    
    #Merge train and valid set together to compute total training set
    #x set
    x_set = np.vstack([x_train, x_valid])
    #y set
    y_set = np.vstack([y_train, y_valid])   
    
    #Iterate over each test point
    for pt_test in x_test:
        #Compute the distances between x set train and test points
        x_dist_list = []
        for d in range(0, len(x_set)):
            x_dist = dist_met(x_set[d], pt_test)
            x_dist_list.append((x_dist, y_set[d]))
        
        #Sort the x distance list from smallest to largest distance
        x_dist_list.sort(key=lambda x: x[0])
        
        #Compute the average for k nearest y values
        new_dist_list = x_dist_list[:k]
        y_tot = []
        #Add in all y values to the y_tot list
        for y_dist in new_dist_list:
            y_tot.append(y_dist[1])
        #Compute the average for the y values
        y_avg = sum(y_tot)/len(y_tot)
        #Add the average value for y into the prediction list
        pred.append(y_avg)
        
    #Compute the RMSE values
    error_test = rmse(pred, y_test)
    
    #Only plot for the dataset mauna_loa
    if plot:
        plt.figure(2)
        plt.plot(x_test, y_test, 'bo-', label='Actual Values', linewidth=2)
        plt.plot(x_test, pred, 'yo-', label='Prediction Values', linewidth=2)
        plt.legend(loc = 'upper left')
        plt.xlabel('x_test')
        plt.ylabel('y')
        plt.title('Mauna Loa Dataset - Test Predictions')
        plt.savefig('mauna_loa_pred.png')
    
    return error_test

#One-Fold Classification Function 
def of_class(x_train, y_train, x_valid, y_valid, dist_met, k_list = None):
    #Variables Initialization
    #A dict using k and the distance metrics as keys, to store the correctness for all total points
    corr = {}
    #Ratio List to store the final ratios of correctness
    ratio_final = []
    
    #Check if user inputs a k_list
    #If not, iterate through k = 0 to 25 to check for the best k
    #The range of k's can be changed
    if k_list == None:
        k_list = list(range(0,26))
    
    #Computation of distances between points in train and valid sets using distance metrics
    for met in dist_met:
        # Compute distances according to distance function for one validation point
        for j in range(0, len(x_valid)):
            x_dist_list = []
            for d in range(0, len(x_train)):
                x_dist = met(x_train[d], x_valid[j])
                x_dist_list.append((x_dist, y_train[d]))

            #Sort the x distance list from smallest to largest distance
            x_dist_list.sort(key=lambda x: x[0])

            classes = {}
            #Find k-NN for x_valid[j]
            for k in k_list:
                classes[k] = []
                new_dist_list = x_dist_list[:k + 1]
                #Add in all the according y train values to the classes list
                for y_dist in new_dist_list:
                    classes[k].append(y_dist[1])

            for k in k_list:
                #Find the class that occurs most commonly
                #Occurance dict to keep track of how many times a point occurs
                occ = {}
                for pt in classes[k]:
                    str_pt = str(pt)
                    #Check if data point is already in occurance dict
                    #If not, add the point in with the occurance time 0, and with the dict key being the point
                    if str_pt not in occ:
                        occ[str_pt] = (pt, 0)
                    #If data point is already in the dict (i.e. there has been at least one occurance of the same class/data point)
                    #Then, add 1 to the occurance time
                    occ[str_pt] = (pt, occ[str_pt][1]+1)

                #Extract all the values from the occurance dict 
                occ_list = list(occ.values())
                #Sort the list by the occurance times, largest to smallest
                occ_list.sort(key=lambda x: x[1], reverse=True)

                #The estimation value is the y value that occured the most times
                estim = occ_list[0][0]
                
                if np.all(estim == y_valid[j]):
                    #If the estimation value is equal to the y_valid value
                    #Check if there is already a value in the corr dict with key (k+1, dist_met)
                    if (k+1, met) not in corr:
                        #If not, set one and equate the value to 0
                        corr[(k+1, met)] = 0
                    #If there is, add 1 to original value
                    corr[(k+1, met)] += 1

    #For each k after the 5 folds, compute the ratio of correctness
    for k, met in corr:
        #Compute the accuracy percentage
        ratio = corr[(k, met)]/len(y_valid)
        #Add the ratio to the final list of ratios with its k and the distance metric used
        ratio_final.append((k, met, ratio))

    return ratio_final

def class_test(x_train, x_valid, x_test, y_train, y_valid, y_test, k, dist_met):
    #Variables Initialization
    #An int to compute the total number of occurance times
    occ_time = 0
    #An int to store the final accuracy percentage
    ratio_final = 0
    
    #Merge train and valid set together to compute total training set
    #x set
    x_set = np.vstack([x_train, x_valid])
    #y set
    y_set = np.vstack([y_train, y_valid])   
    
    #Iterate over each test point
    for j in range(0, len(x_test)):
        #Compute the distances between x set train and test points
        x_dist_list = []
        for d in range(0, len(x_set)):
            x_dist = dist_met(x_set[d], x_test[j])
            x_dist_list.append((x_dist, y_set[d]))
        
        #Sort the x distance list from smallest to largest distance
        x_dist_list.sort(key=lambda x: x[0])
           
        #Compute the occurance times
        new_dist_list = x_dist_list[:k]
        #Occurance dict to keep track of how many times a point occurs
        occ = {}
        #Add in all y values to the y_tot list
        for y_dist in new_dist_list:
            y_dist = y_dist[1]
            str_y_dist = str(y_dist)
            #Check if data point is already in occurance dict
            #If not, add the point in with the occurance time 0, and with the dict key being the point
            if str_y_dist not in occ:
                occ[str_y_dist] = (y_dist, 0)
            #If data point is already in the dict (i.e. there has been at least one occurance of the same class/data point)
            #Then, add 1 to the occurance time
            occ[str_y_dist] = (y_dist, occ[str_y_dist][1]+1)
            
        #Extract all the values from the occurance dict
        occ_list = list(occ.values())
        #Sort the list by the occurance times
        occ_list.sort(key=lambda x: x[1], reverse = True)
            
        #The estimation value is the y value that occured the most times
        estim = occ_list[0][0]
            
        #Check how many times the estimation value is actually equal to each y_valid value
        if np.all(estim == y_test[j]):
            occ_time += 1
    
        #Compute the accuracy percentage
        ratio_final = occ_time/len(x_test)
        
    return ratio_final    
    
def train_model(dataset, model):
    #A list of the 3 different types of distance metrics
    dist_met = [minkowski_1, minkowski_2]
    
    #Training for Regression Model
    if model == 'regression':
        #Only if dataset is rosenbrock, use n_train=100, d=2
        if dataset == 'rosenbrock':
            x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset, n_train = 1000, d = 2)
        else:
            x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)
        
        if dataset == 'mauna_loa':
            #error_final is a list of (k, distance metric, average RMSE value) as values
            #Use l_2 (i.e. Minkowski_2) for distance metric
            error_final = ff_cross_valid(x_train, x_valid, y_train, y_valid,[minkowski_2])
            
            #Sort the list by smallest to largest k value
            error_final.sort(key=lambda x: x[0])
            
            #Compute the k values list and the RMSE values list
            k_list = []
            err_list = []
            for k, met, error in error_final:
                k_list.append(k)
                err_list.append(error)
            
            #Plot
            plt.figure(1)
            plt.plot(k_list, err_list, '-r')
            plt.xlabel('k')
            plt.ylabel('Five-Fold Cross Validation Average RMSE')
            plt.title('Mauna Loa - Five-Fold Cross Validation RMSE VS various k values')
            plt.savefig('mauna_loa_ff_cross_valid_minkowski_2.png')
            
            #Sort the list by smallest to largest average RMSE value
            error_final.sort(key=lambda x: x[2])
            
            #Compute k value and the distance metric that results in the smallest average RMSE value
            k_min_err = error_final[0][0]
            met_min_err = error_final[0][1]
            min_rmse = error_final[0][2]
            
            #error_test is a list of the RMSE value of the predicted values and the y_test values
            error_test = regression_test(x_train, x_valid, x_test, y_train, y_valid, y_test, k_min_err, met_min_err, True)
            
            print('----------Mauna Loa (With l_2 Norm) Results----------')
            print('Optimal k: ' + '\n' + str(k_min_err))
            print('Distance Metric: ' + '\n' + str(met_min_err))
            print('Five Fold Cross Validation RMSE' + '\n' + str(min_rmse))
            print('Test RMSE: ' + '\n' + str(error_test) + '\n')
        
        #For all other datasets that are not mauna_loa
        #error_final is a list of (k, distance metric, average RMSE value) as values
        #Use all three distance metrics
        error_final = ff_cross_valid(x_train, x_valid, y_train, y_valid, dist_met)
            
        #Sort the list by smallest to largest average RMSE value
        error_final.sort(key=lambda x: x[2])
            
        #Compute k value and the distance metric that results in the smallest average RMSE value
        k_min_err = error_final[0][0]
        met_min_err = error_final[0][1]
        min_rmse = error_final[0][2]

        #error_test is a list of the RMSE value of the predicted values and the y_test values
        error_test = regression_test(x_train, x_valid, x_test, y_train, y_valid, y_test, k_min_err, met_min_err, True)
        
        return k_min_err, met_min_err, min_rmse, error_test
    
    #Training for Classification Model
    elif model == 'classification':
        #Load dataset
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)
        
        #ratio_final is a list of (k, distance metric, ratio) values
        ratio_final = of_class(x_train, y_train, x_valid, y_valid, dist_met)
       
        #Sort the list by largest to smallest accuracy ratio
        ratio_final.sort(key=lambda x: x[2], reverse = True)        
        #Compute k value and the distance metric that results in the largest accuracy ratio
        k_max_rat = ratio_final[0][0]
        met_max_rat = ratio_final[0][1]
        max_rat = ratio_final[0][2]
        
        #ratio_test is the accuracy ratio with the testing set
        ratio_test = class_test(x_train, x_valid, x_test, y_train, y_valid, y_test, k_max_rat, met_max_rat)
        
        return k_max_rat, met_max_rat, max_rat, ratio_test
    
    return True     

#Test the Performance of 2 Different Approaches Function
def perform_test(x_set, x_test, y_set, y_test, k, dist_met, approach):
    #Variables Initialization
    #Start Time
    start_time = time.time()
    #Prediction list to store the prediction for each data point
    pred = []
    
    #Brute_Force Approach
    if approach == 'Brute_Force':
        #Iterate over each test point
        for pt_test in x_test:
            #Compute the distances between x set train and test points
            x_dist_list = []
            for d in range(0, len(x_set)):
                x_dist = dist_met(x_set[d], pt_test)
                x_dist_list.append((x_dist, y_set[d]))
        
            #Sort the x distance list from smallest to largest distance
            x_dist_list.sort(key=lambda x: x[0])
        
            #Compute the average for k nearest y values
            new_dist_list = x_dist_list[:k]
            y_tot = []
            #Add in all y values to the y_tot list
            for y_dist in new_dist_list:
                y_tot.append(y_dist[1])
            #Compute the average for the y values
            y_avg = sum(y_tot)/len(y_tot)
            #Add the average value for y into the prediction list
            pred.append(y_avg)
        
        #Compute RMSE value for between the predictions and the testing set
        error_test = rmse(pred, y_test)
    
    #k-d Tree Approach
    elif approach == 'kd_tree':
        kd_t = neighbors.KDTree(x_set)
        #Compute the distances and indices of the k nearest neighbours
        dist, ind = kd_t.query(x_test, k = k)
        #Add the predicted values of y to the list
        pred = np.sum(y_set[ind], axis = 1)/k
        
        #Compute RMSE value for between the predictions and the testing set
        error_test = rmse(pred, y_test)
        
    #Compute total time used to run the approach
    run_time = time.time() - start_time
    
    return run_time, error_test

#Linear Regression Algorithm Function
def lin_reg(x_train, x_valid, x_test, y_train, y_valid, y_test, model):
    #Merge train and valid set together to compute total training set
    #x set
    x_set = np.vstack([x_train, x_valid])
    #y set
    y_set = np.vstack([y_train, y_valid]) 

    #Compute matrix X for SVD
    X = np.ones((len(x_set), len(x_set[0])+1))
    X[:, 1:] = x_set
        
    #Compute matrix X for test set
    X_test = np.ones(len(x_test), len(x_test[0]+1))
    X_test[:, 1:] = x_test
        
    #Compute SVD
    U, S, V = np.linalg.svd(X)
    #Compute Inverted Sigma
    sigma = np.diag(S)
    zeros = np.zeros([len(x_set)-len(S), len(S)])
    sigma_inv = np.linalg.pinv(np.vstack([sigma, zeros]))
        
    #Compute weights
    weight = np.dot(V.T, np.dot(sigma_inv, np.dot(U.T, y_set)))
        
    #Regression Model
    if model == 'regression':
        #Compute Predictions
        pred = np.dot(X_test, weight)
        
        #Compute RMSE value for between the predictions and the testing set
        final = rmse(pred, y_test)
        
    elif model == 'classification':
        #Compute Predictions
        pred = np.argmax(np.dot(X_test, weight), axis = 1)
        y_test = np.argmax(1*y_test, axis = 1)
        
        #Compute Predictions Accuracy Ratio
        corr_sum = (pred == y_test).sum()
        final = corr_sum/len(y_test)
        
    return final
        
#Main Function to run the functions
if __name__ == '__main__':
    #Datasets
    #All sets
    sets_tot = ['mauna_loa', 'rosenbrock', 'pumadyn32nm', 'iris', 'mnist_small']
    #Regression sets
    sets_reg = ['mauna_loa', 'rosenbrock', 'pumadyn32nm']
    #Classification sets
    sets_class = ['iris', 'mnist_small']
    
    #Question 1 Results
    """print('Question 1: ' + '\n')
    for sets in sets_reg:
        k_min_err, met_min_err, min_rmse, error_test = train_model(sets, 'regression')
        print('---------- ' + sets + ' Results----------')
        print('Optimal k: ' + '\n' + str(k_min_err))
        print('Optimal Distance Metric: ' + '\n' + str(met_min_err))
        print('Five Fold Cross Validation RMSE' + '\n' + str(min_rmse))
        print('Test RMSE: ' + '\n' + str(error_test) + '\n')"""
        
    #Question 2 Results
    """print('Question 2: ' + '\n')
    for sets in sets_class:
        k_max_rat, met_max_rat, max_rat, ratio_test = train_model(sets, 'classification')
        print('---------- ' + sets + ' Results----------')
        print('Optimal k: ' + '\n' + str(k_max_rat))
        print('Optimal Distance Metric: ' + '\n' + str(met_max_rat))
        print('Validation Ratio' + '\n' + str(max_rat))
        print('Test Ratio: ' + '\n' + str(ratio_test) + '\n')"""    

    #Question 3 Results
    """print('Question 3: ' + '\n')
    table_final = {}
    table_final['Brute_Force'] = []
    table_final['kd_tree'] = []
    #Iterate over various d values
    for d in range(2,10):
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train = 5000, d = d)
        #Merge train and valid set together to compute total training set
        #x set
        x_set = np.vstack([x_train, x_valid])
        #y set
        y_set = np.vstack([y_train, y_valid])  
        
        #Brute-Force Approach
        run_time, error_test = perform_test(x_set, x_test, y_set, y_test, 5, minkowski_2, 'Brute_Force')
        table_final['Brute_Force'].append((d, run_time, error_test)) 
        #k-d Tree Approach
        run_time, error_test = perform_test(x_set, x_test, y_set, y_test, 5, minkowski_2, 'kd_tree')
        table_final['kd_tree'].append((d, run_time, error_test)) 
    
    app = list(table_final.keys())
    runtime_list = []
    rmse_list = []
    d_list = list(range(2,10))
    for i in range(0, 8):
        print('---------- ' + 'd = ' + str(i+2) + ' Results----------')
        print('Run Times:' + '\n')
        print(app[0] + ': ' + str(table_final[app[0]][i][1]))
        print(app[1] + ': ' + str(table_final[app[1]][i][1]) + '\n')
        
        print('RMSE Values: ' + '\n')
        print(app[0] + ': ' + str(table_final[app[0]][i][2]))
        print(app[1] + ': ' + str(table_final[app[1]][i][2]) + '\n')

    for app in table_final:
        for i in range(0, len(table_final[app])):
            runtime_list.append(table_final[app][i][1])
            rmse_list.append(table_final[app][i][2])
            
    plt.figure(3)
    plt.plot(d_list, runtime_list[0:8], '-b', label = 'Brute Force')
    plt.figure(4)
    plt.plot(d_list, rmse_list[0:8], '-b', label = 'Brute Force')
    
    plt.figure(5)
    plt.plot(d_list, runtime_list[8:16], '-y', label = 'KDTree')
    plt.figure(6)
    plt.plot(d_list, rmse_list[8:16], '-y', label = 'KDTree')
        
    #Graph Labels
    plt.figure(3)
    plt.legend(loc = 'upper right')
    plt.xlabel('d')
    plt.ylabel('Time [s]')
    plt.title('Runtime VS Various d')
    plt.savefig('runtime_vs_various_d.png')
    plt.figure(4)
    plt.legend(loc = 'upper right')
    plt.xlabel('d')
    plt.ylabel('RMSE')
    plt.title('RMSE Values VS Various d')
    plt.savefig('rmse_vs_various_d.png')   
    
    plt.figure(5)
    plt.legend(loc = 'upper right')
    plt.xlabel('d')
    plt.ylabel('Time [s]')
    plt.title('Runtime VS Various d')
    plt.savefig('runtime_vs_various_d.png')
    plt.figure(6)
    plt.legend(loc = 'upper right')
    plt.xlabel('d')
    plt.ylabel('RMSE')
    plt.title('RMSE Values VS Various d')
    plt.savefig('rmse_vs_various_d.png')"""
    
    #Question 4 Results
    """print('Question 4: ' + '\n')  
    #For Regression Sets
    for sets in sets_reg:
        if sets == 'rosenbrock':
            x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(sets, n_train = 1000, d = 2)
        else:
            x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(sets)
        
        rmse_final = lin_reg(x_train, x_valid, x_test, y_train, y_valid, y_test, 'regression')
        print('---------- ' + sets + ' Results----------')
        print('Test RMSE: ' + '\n' + str(rmse_final) + '\n') 
        
    #For Classification Sets
    for sets in sets_class:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(sets)
        
        ratio_final = lin_reg(x_train, x_valid, x_test, y_train, y_valid, y_test, 'classification')
        print('---------- ' + sets + ' Results----------')
        print('Test Ratio: ' + '\n' + str(ratio_final) + '\n')"""
