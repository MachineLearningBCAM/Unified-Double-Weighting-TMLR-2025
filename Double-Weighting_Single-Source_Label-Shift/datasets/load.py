# -*- coding: utf-8 -*-
"""
.. _load:

Set of loaders and convenient functions to access Dataset
=========================================================
"""
import csv
import zipfile
from os.path import dirname, join

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.utils import Bunch


def normalizeLabels(origY):
    """
    Normalize the labels of the instances in the range 0,...r-1 for r classes
    """

    # Map the values of Y from 0 to r-1
    domY = np.unique(origY)
    Y = np.zeros(origY.shape[0], dtype=int)

    for i, y in enumerate(domY):
        Y[origY == y] = i

    return Y


def load_adult(with_info=False):
    """Load and return the adult incomes prediction dataset (classification).

    =================   ==============
    Classes                          2
    Samples per class    [37155,11687]
    Samples total                48882
    Dimensionality                  14
    Features             int, positive
    =================   ==============

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``with_info`` is False

    """
    module_path = dirname(__file__)

    data_file_name = join(module_path, 'data', 'adult.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)
        temp = next(data_file)
        # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 filename=data_file_name)


def load_diabetes(with_info=False):
    """Load and return the Pima Indians Diabetes dataset (classification).

    =================   =====================
    Classes                                 2
    Samples per class               [500,268]
    Samples total                         668
    Dimensionality                          8
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``with_info`` is False

    """
    module_path = dirname(__file__)

    data_file_name = join(module_path, 'data', 'diabetes.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 filename=data_file_name)


def load_redwine(with_info=False):
    """Load and return the Red Wine Dataset (classification).

    =================   =====================
    Classes                                10
    Samples per class            [1599, 4898]
    Samples total                        6497
    Dimensionality                         11
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``with_info`` is False

    """
    module_path = dirname(__file__)


    data_file_name = join(module_path, 'data', 'redwine.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray([np.float64(i) for i in d[:-1]],
                                 dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 filename=data_file_name)


def load_usenet2(with_info=False):
    """Load and return the Vehicle Dataset (classification).

    =================   =====================
    Classes                                 2
    Samples per class              [1000,500]
    Samples total                        1500
    Dimensionality                         99
    Features                              int
    =================   =====================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``with_info`` is False

    """
    module_path = dirname(__file__)


    data_file_name = join(module_path, 'data', 'usenet2.csv')

    dataset = np.genfromtxt(data_file_name, skip_header=1, delimiter=',')
    data = dataset[:, :-1]
    target = dataset[:, -1]
    feature_names = []
    if not with_info:
        return data, normalizeLabels(target)
    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 filename=data_file_name)


def load_credit(with_info=False):
    """Load and return the Credit Approval prediction dataset (classification).

    =================   =====================
    Classes                                 2
    Samples total                         690
    Dimensionality                         15
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``with_info`` is False

    """
    module_path = dirname(__file__)


    data_file_name = join(module_path, 'data', 'credit.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=np.float64)
            except ValueError:
                print(i, d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 filename=data_file_name)


def load_mammographic(with_info=False):
    """Load and return the Mammographic Mass Data Set (classification).

    ================== ============
    Classes                      2
    Samples per class    [516, 445]
    Samples total              961
    Dimensionality               5
    Features                   int
    ================== ============

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of mammographic csv dataset.

    (data, target) : tuple if ``with_info`` is False

    """
    module_path = dirname(__file__)


    data_file_name = join(module_path, 'data', 'mammographic.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=np.float64)
            except ValueError:
                print(i, d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 feature_names=['BI-RADS',
                                'age',
                                'shape',
                                'margin',
                                'density'])


def load_20News_500features(with_info=False):
    """Load and return the  Data Set
    (classification).

    ============================ =============================
    Classes                                                 4
    Samples from training distribution per class   
    Samples from training distribution in total          
    Samples from testing distribution per class    
    Samples from testing distribution in total          
    Dimensionality                                        500  
    Features                                              int
    =========================== ==============================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of satellite csv dataset.

    (data, target) : tuple if ``with_info`` is False

    """
    module_path = dirname(__file__)

    with open(join(module_path, 'data',
                   '20News_Train_500features.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples_Train = int(temp[0])
        n_features_Train = int(temp[1])
        target_names = np.array(temp[2:])
        dataTrain = np.empty((n_samples_Train, n_features_Train))
        targetTrain = np.empty((n_samples_Train, ), dtype=int)

        for i, ir in enumerate(data_file):
            dataTrain[i] = np.asarray(ir[:-1], dtype=np.float64)
            targetTrain[i] = np.asarray(ir[-1], dtype=int)
            
    with open(join(module_path, 'data',
                   '20News_Test_500features.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples_Test = int(temp[0])
        n_features_Test = int(temp[1])
        target_names = np.array(temp[2:])
        dataTest = np.empty((n_samples_Test, n_features_Test))
        targetTest = np.empty((n_samples_Test, ), dtype=int)

        for i, ir in enumerate(data_file):
            dataTest[i] = np.asarray(ir[:-1], dtype=np.float64)
            targetTest[i] = np.asarray(ir[-1], dtype=int)        

    if not with_info:
        return dataTrain, normalizeLabels(targetTrain), \
            dataTest, normalizeLabels(targetTest)

    return Bunch(data=dataTrain, target=normalizeLabels(targetTrain),
                 target_names=target_names,
                 ), \
            Bunch(data=dataTest, target=normalizeLabels(targetTest),
                 target_names=target_names,
                 )
