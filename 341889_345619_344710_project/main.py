import argparse

import numpy as np

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression 
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os
import time
import matplotlib.pyplot as plt
np.random.seed(100)

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors

    ##EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load('features.npz',allow_pickle=True)
        xtrain, xtest, ytrain, ytest, ctrain, ctest =feature_data['xtrain'],feature_data['xtest'],\
        feature_data['ytrain'],feature_data['ytest'],feature_data['ctrain'],feature_data['ctest']

    ##ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path,'dog-small-64')
        xtrain, xtest, ytrain, ytest, ctrain, ctest = load_data(data_dir)

    ##TODO: ctrain and ctest are for regression task. (To be used for Linear Regression and KNN)
    ##TODO: xtrain, xtest, ytrain, ytest are for classification task. (To be used for Logistic Regression and KNN)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        ### WRITE YOUR CODE HERE
        index_x = int(len(xtrain)*0.8)
        xtest = xtrain[index_x:]
        xtrain = xtrain[:index_x]
        ytest = ytrain[index_x:]
        ytrain = ytrain[:index_x]
        ctest = ctrain[index_x:]
        ctrain = ctrain[:index_x]
        pass

    ### WRITE YOUR CODE HERE to do any other data processing
    if args.bias_term :
        xtrain = append_bias_term(xtrain)
        xtest = append_bias_term(xtest)

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)
    elif args.method == "linear_regression": 
        method_obj = LinearRegression(lmda=args.lmda)
    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr,max_iters=args.max_iters)
    elif args.method == "knn":
        if args.task == "center_locating":
            method_obj = KNN(k=args.K, task_kind=args.task)
        else:
             method_obj = KNN(k=args.K)

    ## 4. Train and evaluate the method
    
    if args.task == "center_locating":
        # Fit parameters on training data
        s1 = time.time()
        preds_train = method_obj.fit(xtrain, ctrain)
        s2 = time.time()

        # Perform inference for training and test data
        train_pred = method_obj.predict(xtrain)
        s3 = time.time()
        preds = method_obj.predict(xtest)
        s4 = time.time()    

        ## Report results: performance on train and valid/test sets
        train_loss = mse_fn(train_pred, ctrain)
        loss = mse_fn(preds, ctest)

        print(f"\nTrain loss = {train_loss:.3f}% - Test loss = {loss:.4f}")

    elif args.task == "breed_identifying":
        # Fit (:=train) the method on the training data for classification task
        s1 = time.time()
        preds_train = method_obj.fit(xtrain, ytrain)
        s2 = time.time()

        # Predict on unseen data
        s3 = s2
        preds = method_obj.predict(xtest)
        s4 = time.time() 

        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    else:
        raise Exception("Invalid choice of task! Only support center_locating and breed_identifying!")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    if args.task == "breed_identifying":
        args.acc = acc
    else:
        args.acc = loss
    
    print("Fit time :", s2-s1, "seconds")
    print("Predict time :", s4-s3, "seconds")

if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="center_locating", type=str, help="center_locating / breed_identifying")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
    parser.add_argument('--data_path', default="data", type=str, help="path to your dataset")
    parser.add_argument('--data_type', default="features", type=str, help="features/original(MS2)")
    parser.add_argument('--lmda', '--lamda', type=float, default=10, help="lambda of linear/ridge regression")
    parser.add_argument('--K', type=int, default=1, help="number of neighboring datapoints used for knn")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=500, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # Feel free to add more arguments here if you need!

    parser.add_argument('--bias_term', action="store_true", help="add a bias term to the data")
    parser.add_argument('--test_parameters', action="store_true", help="test different values of parameters for the model")

    # MS2 arguments
    parser.add_argument('--nn_type', default="cnn", help="which network to use, can be 'Transformer' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args() 

    if args.test_parameters:
        fig = plt.figure()
        if args.method == "linear_regression":
            tab = np.arange(0,5,0.1)
            size = len(tab)
            tab_x = np.zeros(size)
            tab_y = np.zeros(size)
            args.bestpair = (0,10000000)

            for x in range(size):
                args.lmda = tab[x]
                tab_x[x] = args.lmda
                main(args)
                tab_y[x] = args.acc 
                if args.acc < args.bestpair[1] :
                    args.bestpair = (args.lmda,args.acc)
            
            print("")
            print(f"Best lmda parameter is {args.bestpair[0]:.3f}")
            plt.xlabel("lmda")
            plt.ylabel("MSE")
            plt.plot(tab_x, tab_y)
        
        if args.method == "logistic_regression":
            args.acc = 0
            #tab = [1e-5,1e-4,1e-3,1e-2,1e-1]
            tab = np.arange(5e-4,55e-4,5e-4)
            tab_max_iters = np.arange(100,301,50)
            size = len(tab)
            tab_x = np.zeros(size)
            tab_y = np.zeros(size)
            args.bestpair = (0,0,0)

            for y in tab_max_iters:
                args.max_iters = y
                for x in range(size):
                    args.lr = tab[x]
                    tab_x[x] = args.lr
                    main(args)
                    tab_y[x] = args.acc
                    if args.acc>=args.bestpair[1]:
                        args.bestpair = (args.lr,args.acc,args.max_iters)
                plt.plot(tab_x, tab_y)
            
            print("")
            print(f"Best lr parameter is {args.bestpair[0]:.4f} and best max_iters parameter is {args.bestpair[2]}")
            plt.plot(args.bestpair[0],args.bestpair[1], marker="o", color="r", label=f"Best parameters \n(max_iters={args.bestpair[2]},accuracy={args.bestpair[1]:.2f})")
            plt.xlabel("lr")
            plt.ylabel("accuracy")
            plt.legend(loc=4)

        if args.method == "knn":
            args.acc = 0
            tab = np.arange(1,101,10)
            size = len(tab)
            tab_x = np.zeros(size)
            tab_y = np.zeros(size)
            args.bestpair = (0,1000000)
            if args.task=="breed_identifying":
                args.bestpair=(0,0)
                tab = np.arange(10)

            for x in range(size):
                args.K = tab[x]+1
                tab_x[x] = args.K
                main(args)
                tab_y[x] = args.acc
                if args.task=="breed_identifying" and args.acc > args.bestpair[1]:
                    args.bestpair = (args.K,args.acc)
                elif args.task=="center_locating" and args.acc < args.bestpair[1]:
                    args.bestpair = (args.K,args.acc)

            plt.plot(tab_x, tab_y)
            label = f"Best parameters \n(K={args.bestpair[0]},lost={args.bestpair[1]:.3f})"
            ylabel = "MSE"
            if args.task=="breed_identifying":
                label = f"Best parameters \n(K={args.bestpair[0]},accuracy={args.bestpair[1]:.3f})"
                ylabel = "accuracy"
            print("")
            print(f"Best K parameter is {args.bestpair[0]}")
            plt.plot(args.bestpair[0],args.bestpair[1],marker="o",color="r",label=label)
            plt.xlabel("K")
            plt.ylabel(ylabel)
            plt.legend(loc=4)
        plt.show()
        plt.close(fig)
    else :
        main(args)

# (Logistic Regression) : Best accuracy = 87.156% , for lr = 0.0045, max_iters 100 with a bias_term / Best accuracy = 77.982% , for lr = 0.001 without bias_term
# (Linear/Ridge Regression) : Best MSE = 0.0046 , for lmda = 1.2 with a bias_term / Best MSE = 0.051, for lmda = 1 without bias_term
# (KNN Classification) : Best accuracy = 88.685% , for K = 6 
# (KNN Regression) : Best MSE = 0.0046 , decrease as K grows but converge to 0.0046 for K>=30
