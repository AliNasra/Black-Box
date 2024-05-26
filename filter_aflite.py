import numpy as np
import os
import pandas as pd
import math
from matplotlib import pyplot as plt
from sklearn import svm, linear_model
from sklearn.datasets import make_circles,make_moons,make_blobs



def accuracy(pred, ref):
    """Compute simple accuracy."""
    assert len(pred) == len(ref)
    correct = sum((pred == ref).astype('int'))
    return correct/len(ref)


def make_train_test_splits(data, feature_set):
    """Randomly split data into train (90%) and test (10%)."""
    n = len(data)
    test_ids = np.random.choice(range(n), n//10, replace=False)
    test = data.loc[test_ids]

    train_ids = np.array(list(set(range(n)).difference(set(test_ids))))
    train = data.loc[train_ids]
    
    train_data = train[feature_set]
    train_label = train['label']
    test_data = test[feature_set]
    test_label = test['label']
        
    return train_data, train_label, test_data, test_label


def train_rbf_simple(train_data, train_label, test_data, test_label):
    rbf = svm.SVC(gamma='auto',
                  # C=1.0,
                  # gamma=0.10,
                  tol=1e-5, 
                  random_state=np.random.randint(0, 100))   
    rbf.fit(train_data, train_label)
    rbf_predicted = rbf.predict(test_data)
    return accuracy(rbf_predicted, test_label.values)


def train_linear_simple(train_data, train_label, test_data, test_label):        
    lin = linear_model.SGDClassifier(max_iter=1000, 
                                     tol=1e-5)
    lin.fit(train_data, train_label)

    lin_predicted = lin.predict(test_data)
    #return accuracy(lin_predicted, test_label.values)
    return lin_predicted,test_label.values


def train_rbf_split(data, feats=['x', 'y', 'c1', 'c2'], num_seeds=1000):
    """
    Splits data into train and test, and then trains an SVM model with an
    RBF kernel on the train, and with the trained model, computes accuracy 
    on the test set. Repeats the above `num_seeds` times, and returns the 
    mean / std of accuracies on the test set (to reduce variance).
    """
    acc = []
    for _ in range(num_seeds):
        train_data, train_label, test_data, test_label = make_train_test_splits(data, feats)
        
        rbf = svm.SVC(gamma='auto', 
                      tol=1e-5, 
                      # C=1.0,
                      # gamma=0.10,
                      random_state=np.random.randint(0, 100))
        rbf.fit(train_data, train_label)
        
        rbf_predicted = rbf.predict(test_data)
        acc.append(accuracy(rbf_predicted, test_label.values))
    return f"RBF    acc: {np.mean(acc):.4f} (+/-{np.std(acc):.2f})"


def train_linear(data, feats=['x', 'y', 'c1', 'c2'], num_seeds=1000):
    """
    Splits data into train and test, and then trains a linear model on the 
    train, and with the trained model, computes accuracy 
    on the test set. Repeats the above `num_seeds` times, and returns the 
    mean / std of accuracies on the test set (to reduce variance).
    """
    acc = []
    for _ in range(num_seeds):
        train_data, train_label, test_data, test_label = make_train_test_splits(data, feats)
        
        lin = linear_model.SGDClassifier(max_iter=1000, 
                                         tol=1e-5)
        lin.fit(train_data, train_label)
        
        lin_predicted = lin.predict(test_data)
        acc.append(accuracy(lin_predicted, test_label.values))
    return f"LINEAR acc: {np.mean(acc):.4f} (+/-{np.std(acc):.2f})"


def filter(input_file_address):
    np.random.seed(0)
    data         = pd.read_csv(input_file_address)
    #x            = data.loc[filtered_ids]
    acc_lin      = []
    acc_rbf      = []
    num_seeds    = 100
    s            = data
    n            = len(s.index)
    target_size  = int(math.floor(0.7 * n))
    col_list     = data.columns.tolist()
    if "id" not in col_list:
        # Add 'id' column with unique identifiers
        s["id"] = range(0, len(s))
    x_param  = col_list[0:len(col_list)-1]
    y        = col_list[-1]
    while len(s.index) > target_size:
        acc_lin  = []
        #acc_rbf  = []
        #print("****************")
        p_i      = {}
        no       = len(s.index)
        for i in s["id"]:
            p_i[i] = {"T":0,"F":0}
        for _ in range(num_seeds):
            #print(k)
            test_ids      = np.random.choice(s["id"].to_numpy(), int(math.floor(no/(10))), replace=False)
            ftest         = data.loc[test_ids]
            train_ids     = list(set(s["id"].to_numpy()).difference(set(test_ids)))
            ftrain        = data.loc[train_ids]
            fxtrain       = ftrain[x_param]
            fytrain       = ftrain[y]
            fxtest        = ftest[x_param]
            fytest        = ftest[y]
            pred,value    = train_linear_simple(fxtrain, fytrain, fxtest, fytest)
            equal         = pred == value
            for i in range(len(list(equal))):
                if equal[i]:
                    p_i[test_ids[i]]["T"] +=1
                else:
                    p_i[test_ids[i]]["F"] +=1
        predictability = {}
        for i in p_i:
            try:
                predictability[i] = p_i[i]["T"]/(p_i[i]["T"]+p_i[i]["F"])
            except:
                predictability[i] = 0
        to_be_eliminated = []
        for i in predictability:
            if predictability[i] > 0.85:
                to_be_eliminated.append(i)
        if len(to_be_eliminated)<no/60:
            break
        s = s[~s['id'].isin(to_be_eliminated)]
        # Reset the index if needed
        s = s.reset_index(drop=True)
        #accuracy_rbf  = train_rbf_simple(fxtrain, fytrain, fxtest, fytest)    
        #acc_rbf.append()     
    return s

    
    