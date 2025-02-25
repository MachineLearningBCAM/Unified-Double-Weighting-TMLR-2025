import numpy as np
import pandas as pd

from dwls import DWLS
from rwls import RWLS
from MRCpy import CMRC
# Import the datasets
from datasets.load import *
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import NearestNeighbors  


def apply_label_shift(X, Y, n_classes, dataName, ratio):

    if n_classes == 2:
        ratio_tr = [0.5, 0.5]
        ratio_te = [1 - ratio, ratio]
        if dataName == 'adult':
            ntr = 500
            nte = 500
        elif dataName == 'diabetes':
            ntr = 100
            nte = 100
        elif dataName == 'mammographic':
            ntr = 100
            nte = 100
        elif dataName == 'usenet2':
            ntr = 250
            nte = 250
        elif dataName == 'credit':
            ntr = 100
            nte = 100
        ntr_y = [round(ntr * ratio_tr[0]), ntr - round(ntr * ratio_tr[0])]
        nte_y = [round(nte * ratio_te[0]), nte - round(nte * ratio_te[0])]
        
              
    elif n_classes > 2:
        # Define ratios for training and testing
        ratio_tr = [1 / n_classes] * np.ones(n_classes)  # Balanced testing
        if n_classes < 4:
            selected_minority_class = np.random.choice(n_classes, n_classes - 1, replace=False)
        else:
            selected_minority_class = np.random.choice(n_classes, n_classes - 2, replace=False)
        ratio_te = np.zeros(n_classes)
        minority_ratio = ratio
        for minority_class in selected_minority_class:
            ratio_te[minority_class] = minority_ratio
        for i in range(n_classes):
            if i not in selected_minority_class:
                ratio_te[i] = (1 - 2 * minority_ratio) / (n_classes - 2) 
        
        if dataName == '20News_500features':
            ntr = 300
            nte = 300
        elif dataName == 'redwine':
            ntr = 100
            nte = 100
    
        ntr_y = [round(ntr * ratio_tr[i]) for i in range(n_classes)]
        nte_y = [round(nte * ratio_te[i]) for i in range(n_classes)]
    
    yTr = []
    yPte = []
    yTe = []    
    for i in range(n_classes):
        yTr.extend([i] * ntr_y[i])
        yPte.extend([i] * nte_y[i])
        yTe.extend([i] * nte_y[i])

    xTr = []
    xPte = []
    xTe = []

    for i in range(n_classes):
        class_indices = np.where(Y == i)[0]
            
        xTr.extend(X[class_indices[:ntr_y[i]]])
        xPte.extend(X[class_indices[ntr_y[i]:ntr_y[i] + nte_y[i]]])
        xTe.extend(X[class_indices[ntr_y[i] + nte_y[i]:ntr_y[i] + 2 * nte_y[i]]])
            
    xTr = np.array(xTr)
    xPte = np.array(xPte)
    xTe = np.array(xTe)

    yTr = np.concatenate([np.full(ntr_y[i], i) for i in range(n_classes)])
    yPte = np.concatenate([np.full(nte_y[i], i) for i in range(n_classes)])
    yTe = np.concatenate([np.full(nte_y[i], i) for i in range(n_classes)])

    return xTr, yTr, xPte, yPte, xTe, yTe

def apply_label_shift_20News(X_train, X_test, Y_train, Y_test, n_classes, dataName, ratio):
        
    ratio_tr = [1 / n_classes] * np.ones(n_classes)  # Balanced testing
    selected_minority_class = np.random.choice(n_classes, 2, replace=False)
    ratio_te = np.zeros(n_classes)
    minority_ratio = ratio
    for minority_class in selected_minority_class:
        ratio_te[minority_class] = minority_ratio
    for i in range(n_classes):
        if i not in selected_minority_class:
            ratio_te[i] = (1 - 2 * minority_ratio) / (n_classes - 2)
        
    if dataName == '20News_500features':
        ntr = 300
        nte = 300
    
    ntr_y = [round(ntr * ratio_tr[i]) for i in range(n_classes)]
    nte_y = [round(nte * ratio_te[i]) for i in range(n_classes)]
    
    yTr = []
    yPte = []
    yTe = []    
    for i in range(n_classes):
        yTr.extend([i] * ntr_y[i])
        yPte.extend([i] * nte_y[i])
        yTe.extend([i] * nte_y[i])

    xTr = []
    xPte = []
    xTe = []

    for i in range(n_classes):
        class_indices_tr = np.where(Y_train == i)[0]
        class_indices_te = np.where(Y_test == i)[0]
            
        xTr.extend(X_train[class_indices_tr[:ntr_y[i]]])

        xTe.extend(X_test[class_indices_te[:nte_y[i]]])
        xPte.extend(X_test[class_indices_tr[nte_y[i]:(2 * nte_y[i])]])
            
    xTr = np.array(xTr)
    xPte = np.array(xPte)
    xTe = np.array(xTe)

    yTr = np.concatenate([np.full(ntr_y[i], i) for i in range(n_classes)])
    yPte = np.concatenate([np.full(nte_y[i], i) for i in range(n_classes)])
    yTe = np.concatenate([np.full(nte_y[i], i) for i in range(n_classes)])

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

        if dataName[j] == '20News_500features':
            if rep < 10:
                xTr, yTr, xPte, yPte, xTe, yTe = apply_label_shift_20News(X_train, X_test, Y_train, Y_test, n_classes, dataName[j], 0.05)
            elif rep < 20:
                xTr, yTr, xPte, yPte, xTe, yTe = apply_label_shift_20News(X_train, X_test, Y_train, Y_test, n_classes, dataName[j], 0.10)
        else:
            if rep < 10:
                xTr, yTr, xPte, yPte, xTe, yTe = apply_label_shift(X_normalized, Y, n_classes, dataName[j],0.05)
            elif rep < 20:
                xTr, yTr, xPte, yPte, xTe, yTe = apply_label_shift(X_normalized, Y, n_classes, dataName[j],0.1)
            
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