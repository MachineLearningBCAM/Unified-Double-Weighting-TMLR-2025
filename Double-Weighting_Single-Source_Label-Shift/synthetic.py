import numpy as np
from scipy.stats import multivariate_normal

from MRCpy import CMRC
from rwls import RWLS
from dwls import DWLS
# Import the datasets
from MRCpy.datasets import *

def generate_samples(ratio_tr, ratio_te, ntr, nte, distribs, nlabels):
    xTr = []
    xTe = []

    if nlabels == 2:
        ntr_y = [round(ntr * ratio_tr[0]), ntr - round(ntr * ratio_tr[0])]
        nte_y = [round(nte * ratio_te[0]), nte - round(nte * ratio_te[0])]
    
    yTr = []
    yTe = []

    for i in range(nlabels):
        yTr.extend([i] * ntr_y[i])
        yTe.extend([i] * nte_y[i])
        xTr.extend(multivariate_normal.rvs(mean=distribs[i]['mu'], cov=distribs[i]['Sigma'], size=ntr_y[i]))
        xTe.extend(multivariate_normal.rvs(mean=distribs[i]['mu'], cov=distribs[i]['Sigma'], size=nte_y[i]))

    return np.array(xTr), np.array(yTr), np.array(xTe), np.array(yTe)

def normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Initialize parameters
ratio_tr_all = np.arange(0.05, 0.55, 0.05)
ratio_te = [0.95, 0.05]
rep_max = 200  # You need to set rep_max value
ntr = 100  # Number of training samples, you need to set it
nte = 100  # Number of testing samples, you need to set it
n_classes = 2  # Number of labels, you need to set it

# Define distribs according to your problem
distribs = [{'mu': [0.5, 0], 'Sigma': [[0.1, 0], [0, 0.1]]},
            {'mu': [-0.5, 0], 'Sigma': [[0.1, 0], [0, 0.1]]}]  # Example distributions

ptr_y = np.zeros(n_classes)
pte_y = np.zeros(n_classes)

Error1 = np.zeros((len(ratio_tr_all),rep_max))
Error2 = np.zeros((len(ratio_tr_all),rep_max))
Error3 = np.zeros((len(ratio_tr_all),rep_max))

for delta in range(len(ratio_tr_all)):
    print("Using delta equal : ", ratio_tr_all[delta])
    ratio_tr = [ratio_tr_all[delta], 1 - ratio_tr_all[delta]]

    for rep in range(rep_max):
        xTr, yTr, xTe, yTe = generate_samples(ratio_tr, ratio_te, ntr, nte, distribs, n_classes)
        xNorm = normalize(np.vstack((xTr, xTe)))
        xTr = xNorm[:ntr, :]
        xTe = xNorm[ntr:ntr + nte, :]

        for i in range(n_classes):
            ptr_y[i] = np.sum(yTr == i) / ntr
        for i in range(n_classes):
            pte_y[i] = np.sum(yTe == i) / nte

        #No. Adapt
        clf = CMRC(loss = 'log', phi = 'linear', fit_intercept = True, s = 0, deterministic=False)
        clf.fit(xTr, yTr, xTe)
        Error1[delta,rep] = clf.error(xTe, yTe)

        #Reweighted
        clf2 = RWLS(loss = 'log', phi = 'linear', weights_beta  = pte_y / ptr_y, deterministic=False)
        clf2.fit(xTr, yTr, xTe)
        Error2[delta,rep] = clf2.error(xTe, yTe)

        #DWLS
        Ds = 1 / (1-np.arange(0, 1, 0.1))**2
        Cs = np.max(pte_y / ptr_y) / np.sqrt(Ds)
        n_Cs = len(Cs)
        RU = np.zeros(n_Cs)
        for i in range(n_Cs):
            beta_ = np.minimum(pte_y / ptr_y, Cs[i] * np.ones(n_classes))          
            alpha_ = np.minimum(Cs[i] * ptr_y / pte_y, np.ones(n_classes))
            clf3 = DWLS(loss = '0-1', phi = 'linear', weights_alpha = alpha_, weights_beta = beta_, deterministic=False)
            clf3.fit(xTr, yTr, xTe)
            RU[i] = clf3.upper_
        j = np.argmax(RU)
        beta_ = np.minimum(pte_y / ptr_y, Cs[j] * np.ones(n_classes))
        alpha_ = np.minimum(Cs[j] * ptr_y / pte_y, np.ones(n_classes))
        clf3 = DWLS(loss = 'log', phi = 'linear', weights_alpha = alpha_, weights_beta = beta_, deterministic=False)
        clf3.fit(xTr, yTr, xTe)            
        Error3[delta,rep] = clf3.error(xTe, yTe)
        
    print("Mean Error of No Adapt.:", np.mean(Error1[delta,:]))
    print("Mean Error of Reweighted:", np.mean(Error2[delta,:]))
    print("Mean Error of Double-Weighting:", np.mean(Error3[delta,:]))


print("Mean Error of No Adapt.:", np.mean(Error1, axis=1))
print("Mean Error of Reweighted:", np.mean(Error2, axis=1))
print("Mean Error of Double-Weighting:", np.mean(Error3, axis=1))