import numpy as np
import cvxpy as cvx
import scipy.special as scs
import warnings
from sklearn.utils import check_X_y, check_array
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import eig, solve
from scipy.optimize import minimize


from numpy.linalg import eig, solve

# Import the MRC super class
from MRCpy import CMRC
from MRCpy.solvers.cvx import *
from MRCpy.solvers.adam import *
from MRCpy.solvers.sgd import *
from MRCpy.solvers.nesterov import *
from MRCpy.phi import \
    BasePhi, \
    RandomFourierPhi, \
    RandomReLUPhi, \
    ThresholdPhi

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RWLS(CMRC):
    '''

    Parameters
    ----------
    loss : `str` {'0-1', 'log'}, default = '0-1'
        Type of loss function to use for the risk minimization. 0-1 loss
        quantifies the probability of classification error at a certain example
        for a certain rule. Log-loss quantifies the minus log-likelihood at a
        certain example for a certain rule.

    deterministic : `bool`, default = `True`
       Whether the prediction of the labels
       should be done in a deterministic way (given a fixed `random_state`
       in the case of using random Fourier or random ReLU features).

    random_state : `int`, RandomState instance, default = `None`
        Random seed used when 'fourier' and 'relu' options for feature mappings
        are used to produce the random weights.

    fit_intercept : `bool`, default = `False`
            Whether to calculate the intercept for MRCs
            If set to false, no intercept will be used in calculations
            (i.e. data is expected to be already centered).

    solver : {?cvx?, 'grad', 'adam'}, default = ?adam?
        Method to use in solving the optimization problem. 
        Default is ?cvx?. To choose a solver,
        you might want to consider the following aspects:

        ?cvx?
            Solves the optimization problem using the CVXPY library.
            Obtains an accurate solution while requiring more time
            than the other methods. 
            Note that the library uses the GUROBI solver in CVXpy for which
            one might need to request for a license.
            A free license can be requested `here 
            <https://www.gurobi.com/academia/academic-program-and-licenses/>`_

        ?grad?
            Solves the optimization using stochastic gradient descent.
            The parameters `max_iters`, `stepsize` and `mini_batch_size`
            determine the number of iterations, the learning rate and
            the batch size for gradient computation respectively.
            Note that the implementation uses nesterov's gradient descent
            in case of ReLU and threshold features, and the above parameters
            do no affect the optimization in this case.

        ?adam?
            Solves the optimization using
            stochastic gradient descent with adam (adam optimizer).
            The parameters `max_iters`, `alpha` and `mini_batch_size`
            determine the number of iterations, the learning rate and
            the batch size for gradient computation respectively.
            Note that the implementation uses nesterov's gradient descent
            in case of ReLU and threshold features, and the above parameters
            do no affect the optimization in this case.
    
    alpha : `float`, default = `0.001`
        Learning rate for ?adam? solver.

    mini_batch_size : `int`, default = `1` or `32`
        The size of the batch to be used for computing the gradient
        in case of stochastic gradient descent and adam optimizer.
        In case of stochastic gradient descent, the default is 1, and
        in case of adam optimizer, the default is 32.

    max_iters : `int`, default = `100000` or `5000` or `2000`
        The maximum number of iterations to use in case of
        ?grad? or ?adam? solver.
        The default value is
        100000 for ?grad? solver and
        5000 for ?adam? solver and 
        2000 for nesterov's gradient descent.   


    weights_beta : `beta`, default = `None`
        Weights beta(y) associated to each label.
    

    phi : `str` or `BasePhi` instance, default = 'linear'
        Type of feature mapping function to use for mapping the input data.
        The currenlty available feature mapping methods are
        'fourier', 'relu', and 'linear'.
        The users can also implement their own feature mapping object
        (should be a `BasePhi` instance) and pass it to this argument.
        Note that when using 'fourier' feature mapping,
        training and testing instances are expected to be normalized.
        To implement a feature mapping, please go through the
        :ref:`Feature Mapping` section.

        'linear'
            It uses the identity feature map referred to as Linear feature map.
            See class `BasePhi`.

        'fourier'
            It uses Random Fourier Feature map. See class `RandomFourierPhi`.

        'relu'
            It uses Rectified Linear Unit (ReLU) features.
            See class `RandomReLUPhi`.

    **phi_kwargs : Additional parameters for feature mappings.
                Groups the multiple optional parameters
                for the corresponding feature mappings(`phi`).

                For example in case of fourier features,
                the number of features is given by `n_components`
                parameter which can be passed as argument -
                `DWGCS(loss='log', phi='fourier', n_components=300)`

                The list of arguments for each feature mappings class
                can be found in the corresponding documentation.
    '''

    def __init__(self,
                 loss='0-1',
                 deterministic=False,
                 random_state=None,
                 fit_intercept=True,
                 beta_method='KMM',
                 lambda_kernel=0.001,
                 lambda_reg=1,
                 solver='adam',
                 alpha=0.01,
                 stepsize='decay',
                 mini_batch_size=None,
                 max_iters=None,
                 weights_beta=None,
                 phi='linear',
                 **phi_kwargs):
        self.beta_ = weights_beta
        self.beta_method = beta_method
        self.lambda_kernel = lambda_kernel
        self.lambda_reg = lambda_reg
        super().__init__(loss,
                         None,
                         deterministic,
                         random_state,
                         fit_intercept,
                         solver,
                         alpha,
                         stepsize,
                         mini_batch_size,
                         max_iters,
                         phi,
                         **phi_kwargs)
    
    def fit(self, xTr, yTr, xTe=None):
        # print("Running latest version")
        '''
        Fit the MRC model.

        Computes the parameters required for the minimax risk optimization
        and then calls the `minimax_risk` function to solve the optimization.

        Parameters
        ----------
        xTr : `array`-like of shape (`n_samples`, `n_dimensions`)
            Training instances used in

            - Calculating the expectation estimates
              that constrain the uncertainty set
              for the minimax risk classification
            - Solving the minimax risk optimization problem.

            `n_samples` is the number of training samples and
            `n_dimensions` is the number of features.

        yTr : `array`-like of shape (`n_samples`, 1), default = `None`
            Labels corresponding to the training instances
            used only to compute the expectation estimates.

        xTe : array-like of shape (`n_samples2`, `n_dimensions`), default = None
            These instances will be used in the minimax risk optimization.
            These extra instances are generally a smaller set and
            give an advantage in training time.

        Returns
        -------
        self :
            Fitted estimator
        '''

        xTr, yTr = check_X_y(xTr, yTr, accept_sparse=True)

        # Check if separate instances are given for the optimization
        if xTe is None:
            raise ValueError('Missing instances from testing distribution ... ')
        else:
            xTe = check_array(xTe, accept_sparse=True)

        # Obtaining the number of classes and mapping the labels to integers
        origY = yTr
        self.classes_ = np.unique(origY)
        n_classes = len(self.classes_)
        yTr = np.zeros(origY.shape[0], dtype=int)

        # Map the values of Y from 0 to n_classes-1
        for i, y in enumerate(self.classes_):
            yTr[origY == y] = i

        # Feature mappings
        if self.phi == 'fourier':
            self.phi = RandomFourierPhi(n_classes=n_classes,
                                        fit_intercept=self.fit_intercept,
                                        random_state=self.random_state,
                                        **self.phi_kwargs)
        elif self.phi == 'linear':
            self.phi = BasePhi(n_classes=n_classes,
                               fit_intercept=self.fit_intercept,
                               **self.phi_kwargs)
        elif self.phi == 'relu':
            self.phi = RandomReLUPhi(n_classes=n_classes,
                                     fit_intercept=self.fit_intercept,
                                     random_state=self.random_state,
                                     **self.phi_kwargs)
        elif not isinstance(self.phi, BasePhi):
            raise ValueError('Unexpected feature mapping type ... ')

        # Fit the feature mappings
        self.phi.fit(xTr, yTr)
        
        if self.beta_ is None:

            # Compute weights alpha and beta
            #print("Computing weights")
            if self.beta_method == 'KMM':
                self.LSKMM(xTr, yTr, xTe, n_classes)
            elif self.beta_method == 'BBSE':
                xWeights, xValid, yWeights, yValid = train_test_split(xTr, yTr, test_size=0.5, random_state=42)
                self.BBSE(xWeights, yWeights, xValid, yValid, xTe, n_classes)
            elif self.beta_method == 'RLLS':
                xWeights, xValid, yWeights, yValid = train_test_split(xTr, yTr, test_size=0.5, random_state=42)
                self.RLLS(xWeights, yWeights, xValid, yValid, xTe, n_classes)
            elif self.beta_method == 'MLLS':
                xWeights, xValid, yWeights, yValid = train_test_split(xTr, yTr, test_size=0.5, random_state=42)
                self.MLLS(xWeights, yWeights, xValid, yValid, xTe, n_classes)
            else:
                raise TypeError("Just TarS, BBSE and RLLS is implemented for now.") 
                               

            self.beta_ = self.pte_ptr
            self.alpha_ = np.ones((n_classes,1))

            # Compute the expectation estimates
            tau_ = self.compute_tau(xTr, yTr)
            lambda_ = np.zeros_like(tau_)    

            # Fit the MRC classifier
            self.gamma_ = self.beta_[yTr.astype(int), 0]
            self.minimax_risk(xTr, tau_, lambda_, n_classes)

        elif self.beta_ is not None:
            self.beta_ = np.reshape(self.beta_, (n_classes, 1))
            self.alpha_ = np.ones((n_classes,1))            

            tau_ = self.compute_tau(xTr, yTr)
            lambda_ = np.zeros_like(tau_)

            # Fit the MRC classifier
            self.gamma_ = self.beta_[yTr, 0]
            self.minimax_risk(xTr, tau_, lambda_, n_classes)
    
        return self
    
    def LSKMM(self, xTr, yTr, xTe, n_classes):
        '''
        Obtain training and testing weights.

        Computes the weights associated to the
        training and testing samples solving the DW-KMM problem.

        Parameters
        ----------
        xTr : `array`-like of shape (`n_samples`, `n_dimensions`)
            Training instances used in

            - Computing the training weights beta and testing weights alpha.

            `n_samples` is the number of training samples and
            `n_dimensions` is the number of features.

        xTr : `array`-like of shape (`n_samples`, `n_dimensions`)
            Testing instances used in

            - Computing the training weights beta and testing weights alpha.

            `n_samples` is the number of training samples and
            `n_dimensions` is the number of features.

        Returns
        -------
        self :
            Weights self.beta_ and self.alpha_
        '''
        n = xTr.shape[0]
        t = xTe.shape[0]
        d = xTr.shape[1]

        if n <= 200:
            sigma_ = 0.8 * np.sqrt(d)
        elif n > 200 and n <= 1200:
            sigma_ = 0.5 * np.sqrt(d)
        elif n > 1200:
            sigma_ = 0.3 * np.sqrt(d)

        B = 10
        epsilon = B / (4 * np.sqrt(n))
        lambda_kernel = self.lambda_kernel

        Kc = np.zeros((t, n))
        Kx = np.zeros((n, n))
        Ky = np.zeros((n, n))

        for i in range(n):
            Kx[i, i] = 1 / 2
            Ky[i, :] = (yTr[i] == yTr).astype(int)
            for j in range(i+1, n):
                Kx[i, j] = np.exp(-np.linalg.norm(xTr[i, :] - xTr[j, :])**2 / (2 * sigma_**2))
            for j in range(t):
                Kc[j, i] = np.exp(-np.linalg.norm(xTe[j, :] - xTr[i, :])**2 / (2 * sigma_**2))

        Kx = Kx + Kx.T

        R = []
        for i in range(n_classes):
            R.append((yTr == i).astype(int))
        R = np.array(R).T

        M2 = np.linalg.inv(Ky + lambda_kernel * np.eye(n)) @ Ky
        A = M2 @ Kx @ M2
        M = np.ones((1, t)) @ Kc @ M2

        pte_ptr = cvx.Variable((n_classes, 1))
        objective = cvx.Minimize(0.5 * cvx.quad_form(pte_ptr, R.T @ A @ R) - (n / t) * M @ R @ pte_ptr)
        constraints = [
            np.ones((1, n)) @ R @ pte_ptr - n <= n * epsilon,
            n - np.ones((1, n)) @ R @ pte_ptr <= n * epsilon,
            R @ pte_ptr >= 0,
            R @ pte_ptr <= B
        ]

        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=cvx.MOSEK, verbose=False)

        self.pte_ptr = pte_ptr.value
        self.min_KMM = prob.value

        return self
    
    def BBSE(self,xTr, yTr, xValid, yValid, xTe, n_classes):
        # Convert numpy arrays to PyTorch tensors
        xTr = torch.tensor(xTr, dtype=torch.float32)
        yTr = torch.tensor(yTr, dtype=torch.long)  # Labels as long for CrossEntropyLoss
        xValid = torch.tensor(xValid, dtype=torch.float32)
        yValid = torch.tensor(yValid, dtype=torch.long)
        xTe = torch.tensor(xTe, dtype=torch.float32)
        
        # Define the size of the input and output
        input_size = xTr.shape[1]
        
        # Create the model
        model = MLP(input_size, n_classes)
        
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # Equivalent to SparseCategoricalCrossentropy
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # Train the model
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(xTr)
            loss = criterion(outputs, yTr)
            loss.backward()
            optimizer.step()
        
        # Predict using the trained model
        model.eval()
        with torch.no_grad():
            zValid = model(xValid).numpy()
            zTe = model(xTe).numpy()
            probabilities_zValid = torch.nn.functional.softmax(model(xValid), dim=1).numpy()
            probabilities_zTe = torch.nn.functional.softmax(model(xTe), dim=1).numpy()

        # Process predictions
        n = len(zValid)
        t = xTe.shape[0]
        
        pte_z = np.zeros(n_classes)
        for i in range(n_classes):
            pte_z[i] = np.sum(np.argmax(zTe, axis=1) == i) / t
        
        C_zy = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                C_zy[i, j] = np.sum((np.argmax(zValid, axis=1) == i) & (yValid.numpy() == j)) / n
        
        if np.min(eig(C_zy)[0]) <= np.random.rand() / n_classes:
            weights = np.ones(n_classes)
        else:
            weights = solve(C_zy, pte_z)
        
        weights = np.maximum(weights, np.zeros(n_classes))
        self.pte_ptr = weights.reshape(-1, 1)
        
        return self

    def RLLS(self, xTr, yTr, xValid, yValid, xTe, n_classes):
        # Convert numpy arrays to PyTorch tensors
        xTr = torch.tensor(xTr, dtype=torch.float32)
        yTr = torch.tensor(yTr, dtype=torch.long)  # Labels as long for CrossEntropyLoss
        xValid = torch.tensor(xValid, dtype=torch.float32)
        yValid = torch.tensor(yValid, dtype=torch.long)
        xTe = torch.tensor(xTe, dtype=torch.float32)
        
        # Define the size of the input and output
        input_size = xTr.shape[1]
        
        # Create the model
        model = MLP(input_size, n_classes)
        
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # Equivalent to SparseCategoricalCrossentropy
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # Train the model
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(xTr)
            loss = criterion(outputs, yTr)
            loss.backward()
            optimizer.step()
        
        # Predict using the trained model
        model.eval()
        with torch.no_grad():
            zValid = model(xValid).numpy()
            zTe = model(xTe).numpy()
            probabilities_zValid = torch.nn.functional.softmax(model(xValid), dim=1).numpy()
            probabilities_zTe = torch.nn.functional.softmax(model(xTe), dim=1).numpy()

        # Process predictions
        n = len(zValid)
        t = xTe.shape[0]
        
        pte_z = np.zeros(n_classes)
        for i in range(n_classes):
            pte_z[i] = np.sum(np.argmax(zTe, axis=1) == i) / t
        
        C_zy = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                C_zy[i, j] = np.sum((np.argmax(zValid, axis=1) == i) & (yValid.numpy() == j)) / n
        
        b = pte_z - np.dot(C_zy, np.ones((n_classes, 1)))
        rho = 3 * (2 * np.log(2 * n_classes / 0.05) / (3 * n) + np.sqrt(2 * np.log(2 * n_classes / 0.05) / n))
        rho = 0.0001 * rho
        
        # Solve the optimization problem using CVXPY
        theta_ = cvx.Variable((n_classes, 1))
        objective = cvx.Minimize(cvx.norm(C_zy @ theta_ - b) + rho * cvx.norm(theta_))
        constraints = [theta_ >= -1]
        prob = cvx.Problem(objective, constraints)
        prob.solve()
        
        weights = 1 + theta_.value * self.lambda_reg
        weights = np.maximum(weights, np.zeros((n_classes, 1)))
        
        self.pte_ptr = weights.reshape(-1, 1)
        
        return self

    def compute_tau(self, xTr, yTr):
        '''
        Compute mean estimate tau using the given training instances.

        Parameters
        ----------
        xTr : `array`-like of shape (`n_samples`, `n_dimensions`)
            Training instances used for solving
            the minimax risk optimization problem.

        yTr : `array`-like of shape (`n_samples`, 1), default = `None`
            Labels corresponding to the training instances
            used only to compute the expectation estimates.

        Returns
        -------
        tau_ :
            Mean expectation estimate
        '''

        phiMatrix = self.phi.eval_xy(xTr, yTr)
        beta_xTr = self.beta_[yTr, 0].reshape(xTr.shape[0], 1)
        phi_betaMatrix = np.multiply(beta_xTr, phiMatrix)
        tau_ = np.mean(phi_betaMatrix, axis = 0)

        return tau_
    
    def compute_lambda(self, xTe, tau_, n_classes):
        '''
        Compute deviation in the mean estimate tau
        using the given testing instances.

        Parameters
        ----------
        xTe : `array`-like of shape (`n_samples`, `n_dimensions`)
            Training instances used for solving
            the minimax risk optimization problem.
        tau_ : `array`-like of shape (`n_features` * `n_classes`)
            The mean estimates
            for the expectations of feature mappings.
        n_classes : `int`
            Number of labels in the dataset.
        
        Returns
        -------
        lambda_ :
            Confidence vector
        '''
        
        d = self.phi.len_
        t = xTe.shape[0]
        delta_ = 1e-6 * np.ones(d)

        # Define the variables of the opt. problem
        lambda_ = cvx.Variable(d)
        p = cvx.Variable((t * n_classes,1))
        # Define the objetive function
        objective = cvx.Minimize(cvx.sum(lambda_))
        # Construct constraints
        phiMatrix = self.phi.eval_x(xTe)        
        alpha_rep = np.reshape(np.repeat(self.alpha_, d),(1, n_classes, d))
        phi_alpha = np.multiply(alpha_rep, phiMatrix)

        phi_alphaMatrix = np.reshape(phi_alpha, (t * n_classes, d))

        # Define the constraints
        constraints = [
            cvx.reshape(tau_ - lambda_ + delta_, (d,1)) <= phi_alphaMatrix.T @ p,
            phi_alphaMatrix.T @ p <= cvx.reshape(tau_ + lambda_ - delta_, (d,1)),
            lambda_ >= np.zeros(d),
            cvx.sum(cvx.reshape(p, (n_classes, t)),axis=0) == np.ones(t) / t,
            p >= np.zeros((t * n_classes, 1))
        ]

        problem = cvx.Problem(objective, constraints)
        try:
            problem.solve(solver = 'ECOS', feastol = 1e-4, reltol = 1e-3, abstol = 1e-4)
        except cvx.error.SolverError:
            try:
                problem.solve(solver = 'SCS')
            except cvx.error.SolverError:
                raise ValueError('CVXpy couldn\'t find a solution for ' + \
                                     'lambda .... ' + \
                         'The problem is ', problem.status)

        lambda_ = np.maximum(lambda_.value, 0)

        return lambda_
    
    def compute_phi(self, X):
        '''
        Compute the feature mapping corresponding to instances given
        for learning the classifiers and prediction.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Instances to be converted to features.

        Returns
        -------
        phi_alpha :
            Feature mapping weighted by alpha
        '''

        d = self.phi.len_

        phiMatrix = self.phi.eval_x(X)
        alpha_rep = np.reshape(np.repeat(self.alpha_, d),(1, len(self.classes_), d))
        phi_alpha = np.multiply(alpha_rep, phiMatrix)

        return phi_alpha
    

    def minimax_risk(self, X, tau_, lambda_, n_classes):
            '''
            Solves the marginally constrained minimax risk
            optimization problem for
            different types of loss (0-1 and log loss).
            When use_cvx=False, it uses SGD optimization for linear and random
            fourier feature mappings and nesterov subgradient approach for
            the rest.

            Parameters
            ----------
            X : `array`-like of shape (`n_samples`, `n_dimensions`)
                Training instances used for solving
                the minimax risk optimization problem.

            tau_ : `array`-like of shape (`n_features` * `n_classes`)
                The mean estimates
                for the expectations of feature mappings.

            lambda_ : `array`-like of shape (`n_features` * `n_classes`)
                The variance in the mean estimates
                for the expectations of the feature mappings.

            n_classes : `int`
                Number of labels in the dataset.

            Returns
            -------
            self :
                Fitted estimator

            '''

            # Set the parameters for the optimization
            self.n_classes = n_classes
            self.tau_ = check_array(tau_, accept_sparse=True, ensure_2d=False)
            self.lambda_ = check_array(lambda_, accept_sparse=True,
                                    ensure_2d=False)
            phi = self.compute_phi(X)

            # Constants
            n = phi.shape[0]
            m = phi.shape[2]

            # Supress the depreciation warnings
            warnings.simplefilter('ignore')

            # In case of 0-1 loss, learn constraints using the phi
            # These constraints are used in the optimization instead of phi

            if self.solver == 'cvx':
                # Use CVXpy for the convex optimization of the MRC.
                raise ValueError('Not implemented!')

            elif self.solver == 'sgd' or self.solver == 'adam':

                if self.loss == '0-1':
                    # Function to calculate the psi subobjective
                    # to be added to the objective function.
                    # In addition the function returns subgradient
                    # of the expected value psi
                    # to be used by nesterov optimization.
                    def f_(mu):
                        # First we calculate the all possible values of psi
                        # for all the points.

                        psi = 0
                        psi_grad = np.zeros(phi.shape[2], dtype=np.float64)

                        for i in range(n):
                            # Get psi for each data point
                            # and return the max value over all subset
                            # and its corresponding index.
                            g, psi_xi = self.psi(mu, phi[i, :, :])
                            g = self.gamma_[i] * g
                            psi_xi = self.gamma_[i] * psi_xi
                            psi_grad = psi_grad + g
                            psi = psi + psi_xi

                        psi = ((1 / n) * psi)
                        psi_grad = ((1 / n) * psi_grad)
                        return psi, psi_grad

                    # When using SGD for the convex optimization
                    # To compute the subgradient of the subobjective at one point
                    def g_(mu, batch_start_sample_id, batch_end_sample_id, n):
                        i = batch_start_sample_id
                        psi = 0
                        psi_grad = np.zeros(phi.shape[2], dtype=np.float64)
                        while i < batch_end_sample_id:
                            sample_id = i % n
                            g, psi_xi = self.psi(mu, phi[sample_id, :, :])
                            g = self.gamma_[sample_id] * g
                            psi_xi = self.gamma_[sample_id] * psi_xi
                            psi_grad = psi_grad + g
                            psi = psi + psi_xi
                            i = i + 1

                        batch_size = batch_end_sample_id - batch_start_sample_id
                        psi_grad = ((1 / batch_size) * psi_grad)
                        psi = ((1 / batch_size) * psi)
                        return psi_grad

                elif self.loss == 'log':
                    # Define the objective function and
                    # the gradient for the log loss function.

                    # The psi subobjective for all the datapoints
                    def f_(mu):
                        phi_mu = phi @ mu
                        psi = (1 / n) *\
                                self.gamma_.T @ scs.logsumexp((phi_mu), axis=1)

                        # Only computed in case of nesterov subgradient.
                        # In case of SGD, not required.
                        psi_grad = None

                        return psi, psi_grad

                    # Use SGD for the convex optimization in general.
                    # Gradient of the subobjective (psi) at an instance.
                    def g_(mu, batch_start_sample_id, batch_end_sample_id, n):
                        i = batch_start_sample_id
                        expPhi = 0
                        batch_size = batch_end_sample_id - batch_start_sample_id
                        while i < batch_end_sample_id:
                            sample_id = i % n

                            expPhi_xi = np.exp(phi[sample_id, :, :] @ mu
                                            )[np.newaxis, np.newaxis, :]

                            sumExpPhi_xi = \
                                    np.sum(((expPhi_xi @ phi[sample_id, :, :])
                                            [:, 0, :] /
                                            np.sum(expPhi_xi, axis=2)).transpose(),
                                        axis=1)

                            expPhi = expPhi + self.gamma_[sample_id] * sumExpPhi_xi

                            i = i + 1

                        expPhi = ((1 / batch_size) * expPhi)
                        return expPhi

                if isinstance(self.phi, RandomReLUPhi) or \
                isinstance(self.phi, ThresholdPhi):
                    self.params_ = nesterov_optimization_cmrc(self.tau_,
                                                            self.lambda_,
                                                            m,
                                                            f_,
                                                            None,
                                                            self.max_iters)
                elif self.solver == 'sgd':
                    self.params_ = SGD_optimization(self.tau_,
                                                    self.lambda_,
                                                    n,
                                                    m,
                                                    f_,
                                                    g_,
                                                    self.max_iters,
                                                    self.stepsize,
                                                    self.mini_batch_size)
                elif self.solver == 'adam':
                    self.params_ = adam(self.tau_,
                                        self.lambda_,
                                        n,
                                        m,
                                        f_,
                                        g_,
                                        self.max_iters,
                                        self.alpha,
                                        self.mini_batch_size)

                self.mu_ = self.params_['mu']
                self.upper_ = self.params_['best_value']

            else:
                raise ValueError('Unexpected solver ... ')

            self.is_fitted_ = True

            return self

    def psi(self, mu, phi):
            '''
            Function to compute the psi function in the objective
            using the given solution mu and the feature mapping 
            corresponding to a single instance.

            Parameters:
            -----------
            mu : `array`-like of shape (n_features)
                Solution.

            phi : `array`-like of shape (n_classes, n_features)
                Feature mapping corresponding to an instance and
                each class.

            Returns:
            --------
            g : `array`-like of shape (n_features)
                Gradient of psi for a given solution and feature mapping.

            psi_value : `int`
                The value of psi for a given solution and feature mapping.
            '''

            v = phi @ mu.T
            indices = np.argsort(v)[::-1]
            value = v[indices[0]] - 1
            g = phi[indices[0],:]

            for k in range(1, self.n_classes):
                new_value = (k * value + v[indices[k]]) / (k+1)
                if new_value >= value:
                    value = new_value
                    g = (k * g + phi[indices[k],:]) / (k+1)
                else:
                    break

            return g, (value + 1)

    
    def MLLS(self, xTr, yTr, xValid, yValid, xTe, n_classes):
        """
        Perform Maximum Likelihood Label Shift (MLLS) estimation.

        Args:
            xTr: Training data (N, d).
            yTr: Training labels (N,).
            xValid: Labeled validation data (N, d).
            yValid: Validation labels (N,).
            xTe: Unlabeled test data (M, d).

        Returns:
            Estimated weights w (n_classes,).
        """
        input_size = xTr.shape[1]

        xTr = torch.tensor(xTr, dtype=torch.float32)
        yTr = torch.tensor(yTr, dtype=torch.long)
        xValid = torch.tensor(xValid, dtype=torch.float32)
        yValid = torch.tensor(yValid, dtype=torch.long)
        xTe = torch.tensor(xTe, dtype=torch.float32)

        class_counts = torch.bincount(yTr, minlength=n_classes)
        p_train = class_counts / len(yTr)
        
        model = MLP(input_size, n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(xTr)
            loss = criterion(outputs, yTr)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probabilities_zValid = torch.nn.functional.softmax(model(xValid), dim=1)
            probabilities_zTest  = torch.nn.functional.softmax(model(xTe), dim=1)

        T = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
        b = torch.zeros(n_classes, requires_grad=True, dtype=torch.float32)

        optimizer = torch.optim.Adam([T, b], lr=1e-3)
        
        criterion = nn.MSELoss()

        def calibration_function(probabilities, T, b):
            logits = torch.log(probabilities)  # Convert probabilities to logits
            scaled_logits = logits / T + b  # Scale logits and add bias
            calibrated_probs = torch.softmax(scaled_logits, dim=1)  # Apply softmax
            return calibrated_probs

        for epoch in range(100):  
            optimizer.zero_grad()
            calibrated_probs = calibration_function(probabilities_zValid, T, b)
            true_probs = torch.zeros_like(calibrated_probs)
            true_probs.scatter_(1, yValid.view(-1, 1), 1)
            #loss = criterion(calibrated_probs, true_probs)
            lambda_reg = 1e-5
            loss = criterion(calibrated_probs, true_probs) + lambda_reg * (T**2).sum() + lambda_reg * (b**2).sum()
            loss.backward()
            optimizer.step()

        T = T.detach().item()
        b = b.detach().numpy()


        calibrated_probs = calibration_function(probabilities_zTest, T, b).numpy()

        w = cvx.Variable(n_classes, nonneg=True)
        log_likelihood = -cvx.sum(cvx.log(calibrated_probs @ w)) / calibrated_probs.shape[0]
        constraints = [cvx.abs(cvx.sum(cvx.multiply(p_train, w)) - 1) <= 1e-3]
        # problem = cvx.Problem(cvx.Minimize(log_likelihood), constraints)
        # constraint_penalty = 1e-3 * (cvx.sum(cvx.multiply(p_train, w)) - 1)**2
        problem = cvx.Problem(cvx.Minimize(log_likelihood), constraints)
        problem.solve()
        self.pte_ptr = w.value.reshape(-1, 1)


        return self
