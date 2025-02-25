import numpy as np
import pandas as pd

from dwls import DWLS
from rwls import RWLS
from MRCpy import CMRC
# Import the datasets
from datasets.load import *
from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def limit_samples(x, y, max_samples):
        if len(x) > max_samples:
            x, _, y, _ = train_test_split(x, y, train_size=max_samples, random_state=42, stratify=y)
        return x, y

def draw_equal_classes(X, Y, max_samples_per_class=250):
    unique_classes, class_counts = np.unique(Y, return_counts=True)
    n_classes = len(unique_classes)

    # Determine the maximum number of samples per class
    samples_per_class = min(max_samples_per_class, min(class_counts))

    # Collect indices for each class
    indices = []
    for cls in unique_classes:
        cls_indices = np.where(Y == cls)[0]
        sampled_indices = np.random.choice(cls_indices, samples_per_class, replace=False)
        indices.extend(sampled_indices)

    # Shuffle indices to ensure randomness
    indices = shuffle(np.array(indices))

    # Create the balanced dataset
    X_balanced = X[indices]
    Y_balanced = Y[indices]

    return X_balanced, Y_balanced

def apply_knock_one_shift(X, Y, parameter, target_labels):
        xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=2/3, random_state=42)
        xPte, xTe, yPte, yTe = train_test_split(xTest, yTest, test_size=0.5, random_state=42)

        xTrain, yTrain = draw_equal_classes(xTrain, yTrain, max_samples_per_class=300)
        xPte, yPte = draw_equal_classes(xPte, yPte)
        xTe, yTe = draw_equal_classes(xTe, yTe)

        n_classes = len(np.unique(Y))
        if n_classes == 2:
            indices_target = np.where(yTrain == target_labels)[0]
            num_target = len(indices_target)
            num_knock = int(num_target * parameter)    
            xTr = np.delete(xTrain, indices_target[0:num_knock], 0)
            yTr = np.delete(yTrain, indices_target[0:num_knock])
        else:
            target_labels = np.random.choice(n_classes, int(np.ceil(n_classes / 2)), replace = False) 
            xTr = xTrain
            yTr = yTrain
            for label in target_labels:
                indices_target = np.where(yTr == label)[0]
                num_target = len(indices_target)
                num_knock = int(num_target * parameter)    
                xTr = np.delete(xTr, indices_target[0:num_knock], 0)
                yTr = np.delete(yTr, indices_target[0:num_knock])
        
        return xTr, yTr, xPte, yPte, xTe, yTe

def apply_knock_one_shift_20news(Xtrain, Xtest, Ytrain, Ytest, parameter, target_labels):
        xPte, xTe, yPte, yTe = train_test_split(Xtest, Ytest, test_size=0.5, random_state=42)

        xTrain, yTrain = draw_equal_classes(Xtrain, Ytrain, max_samples_per_class=150)
        xPte, yPte = draw_equal_classes(xPte, yPte, max_samples_per_class=100)
        xTe, yTe = draw_equal_classes(xTe, yTe, max_samples_per_class=100)

        target_labels = np.random.choice(n_classes, int(np.ceil(n_classes / 2)), replace = False) 
        xTr = xTrain
        yTr = yTrain
        for label in target_labels:
            indices_target = np.where(yTr == label)[0]
            num_target = len(indices_target)
            num_knock = int(num_target * parameter)    
            xTr = np.delete(xTr, indices_target[0:num_knock], 0)
            yTr = np.delete(yTr, indices_target[0:num_knock])
        return xTr, yTr, xPte, yPte, xTe, yTe

# Data sets
loaders = [load_adult, load_diabetes, load_mammographic, load_usenet2, load_credit, load_20News_500features, load_redwine]  
dataName = ["adult", "diabetes", "mammographic", "usenet2", "credit", "20News_500features", "redwine"]

rep_max = 20

columns = ['dataset', 'iteration', 'method', 'error']
results = pd.DataFrame(columns=columns)

Error1 = np.zeros((len(dataName),rep_max))
Error2 = np.zeros((len(dataName),rep_max))
Error3 = np.zeros((len(dataName),rep_max))
Error4 = np.zeros((len(dataName),rep_max))
Error5 = np.zeros((len(dataName),rep_max))
Error6 = np.zeros((len(dataName),rep_max))
Error7 = np.zeros((len(dataName),rep_max))
Error8 = np.zeros((len(dataName),rep_max))

for j, load in enumerate(loaders):

    # Loading the dataset
    if dataName[j] == '20News_500features':
        X_train, Y_train, X_test, Y_test = load()
        n_classes = len(np.unique(Y_train))

    elif dataName[j] == 'redwine':
        X, Y = load()
        mask = (Y != 0) & (Y != 1) & (Y != 5)
        X = X[mask]
        Y_filtered = Y[mask]
        # Rearrange classes: 2 -> 0, 3 -> 1, 4 -> 2
        class_mapping = {2: 0, 3: 1, 4: 2}
        Y = np.array([class_mapping[label] for label in Y_filtered])
        n_classes = len(np.unique(Y))

    else:
        X, Y = load()
        n_classes = len(np.unique(Y))
    
   
    for rep in range(rep_max):
        
        if dataName[j] == '20News_500features':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        else:
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X)
            combined_data = X_normalized

        if dataName[j] == '20News_500features':
            xTr, yTr, xPte, yPte, xTe, yTe = apply_knock_one_shift_20news(X_train, X_test, Y_train, Y_test, 0.9, np.random.choice(n_classes))
        else:
            xTr, yTr, xPte, yPte, xTe, yTe = apply_knock_one_shift(X_normalized, Y, 0.9, np.random.choice(n_classes))
                        
        ntr = xTr.shape[0]
        nte = xTe.shape[0]

        ptr_y = np.zeros(n_classes)
        pte_y = np.zeros(n_classes)
        for k in range(n_classes):
            ptr_y[k] = np.sum(yTr == k) / ntr
        for k in range(n_classes):
            pte_y[k] = np.sum(yTe == k) / nte

        #No Adaptation Method
        clf = CMRC(loss = '0-1', phi = 'linear', fit_intercept = True, s = 0, deterministic=True)
        clf.fit(xTr, yTr, xTe)
        Error1[j,rep] = clf.error(xTe, yTe)

        #TarS Method
        clf2 = RWLS(loss = '0-1', phi = 'linear', deterministic=True)
        clf2.fit(xTr, yTr, xPte)
        Error2[j,rep] = clf2.error(xTe, yTe)

        #BBSE Method
        clf3 = RWLS(loss = '0-1', phi = 'linear', beta_method='BBSE', deterministic=True)
        clf3.fit(xTr, yTr, xPte)
        Error3[j,rep] = clf3.error(xTe, yTe)

        #RLLS Method
        clf4 = RWLS(loss = '0-1', phi = 'linear', beta_method='RLLS', deterministic=True)
        clf4.fit(xTr, yTr, xPte)
        Error4[j,rep] = clf4.error(xTe, yTe)

        #MLLS Method
        clf5 = RWLS(loss = '0-1', phi = 'linear', beta_method='MLLS', deterministic=True)
        clf5.fit(xTr, yTr, xPte)
        Error5[j,rep] = clf5.error(xTe, yTe)
   
        #DW-LS Method
        clf6 = DWLS(loss = '0-1', phi = 'linear', deterministic=True)
        clf6.fit(xTr, yTr, xPte)          
        Error6[j,rep] = clf6.error(xTe, yTe)

        #Reweighted Method using Exact Probabilities
        clf7 = RWLS(loss = '0-1', phi = 'linear', weights_beta  = pte_y / ptr_y, deterministic=True)
        clf7.fit(xTr, yTr, xTe)
        Error7[j,rep] = clf7.error(xTe, yTe)

        #DW Method using Exact Probabilities
        Ds = 1 / (1-np.arange(0, 1, 0.1))**2
        Cs = np.max(pte_y / ptr_y) / np.sqrt(Ds)
        n_Cs = len(Cs)
        RU = np.zeros(n_Cs)
        for i in range(n_Cs):
            beta_ = np.minimum(pte_y / ptr_y, Cs[i] * np.ones(n_classes))          
            alpha_ = np.minimum(Cs[i] * ptr_y / pte_y, np.ones(n_classes))
            clf8 = DWLS(loss = '0-1', phi = 'linear', weights_alpha = alpha_, weights_beta = beta_, deterministic=True)
            clf8.fit(xTr, yTr, xPte)
            RU[i] = clf8.upper_
        ii = np.argmax(RU)
        beta_ = np.minimum(pte_y / ptr_y, Cs[ii] * np.ones(n_classes))
        alpha_ = np.minimum(Cs[ii] * ptr_y / pte_y, np.ones(n_classes))
        clf8 = DWLS(loss = '0-1', phi = 'linear', weights_alpha = alpha_, weights_beta = beta_, deterministic=True)
        clf8.fit(xTr, yTr, xTe)            
        Error8[j,rep] = clf8.error(xTe, yTe)

    
        new_row = {'dataset': dataName[j],
                   'iteration' : rep,
                   'method' : '\'No_Adapt.\'',
                   'error': Error1}
        results.loc[len(results)] = new_row
        
        new_row = {'dataset': dataName[j],
                   'iteration' : rep,
                   'method' : '\'Exact_RW\'',
                   'error': Error7}
        results.loc[len(results)] = new_row

        new_row = {'dataset': dataName[j],
                   'iteration' : rep,
                   'method' : '\'Exact_DW\'',
                   'error': Error8}
        results.loc[len(results)] = new_row

        new_row = {'dataset': dataName[j],
                   'iteration' : rep,
                   'method' : '\'TarS\'',
                   'error': Error2}
        results.loc[len(results)] = new_row

        new_row = {'dataset': dataName[j],
                   'iteration' : rep,
                   'method' : '\'BBSE\'',
                   'error': Error3}
        results.loc[len(results)] = new_row

        new_row = {'dataset': dataName[j],
                   'iteration' : rep,
                   'method' : '\'RLLS\'',
                   'error': Error4}
        results.loc[len(results)] = new_row

        new_row = {'dataset': dataName[j],
                   'iteration' : rep,
                   'method' : '\'MLLS\'',
                   'error': Error5}
        results.loc[len(results)] = new_row

        new_row = {'dataset': dataName[j],
                   'iteration' : rep,
                   'method' : '\'DW-LS\'',
                   'error': Error6}
        results.loc[len(results)] = new_row

    print(dataName[j])
    print(f"Mean Error and Std of No Adapt.: {np.mean(Error1[j, :]):.2f} ± {np.std(Error1[j, :]):.2f}")
    print(f"Mean Error and Std of Exact Reweighted: {np.mean(Error7[j, :]):.2f} ± {np.std(Error7[j, :]):.2f}")
    print(f"Mean Error and Std of Exact DW: {np.mean(Error8[j, :]):.2f} ± {np.std(Error8[j, :]):.2f}")
    print(f"Mean Error and Std of TarS: {np.mean(Error2[j, :]):.2f} ± {np.std(Error2[j, :]):.2f}")
    print(f"Mean Error and Std of BBSE: {np.mean(Error3[j, :]):.2f} ± {np.std(Error3[j, :]):.2f}")
    print(f"Mean Error and Std of RLLS: {np.mean(Error4[j, :]):.2f} ± {np.std(Error4[j, :]):.2f}")
    print(f"Mean Error and Std of MLLS: {np.mean(Error5[j, :]):.2f} ± {np.std(Error5[j, :]):.2f}")
    print(f"Mean Error and Std of DW-LS: {np.mean(Error6[j, :]):.2f} ± {np.std(Error6[j, :]):.2f}")