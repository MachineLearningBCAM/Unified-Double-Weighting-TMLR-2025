import numpy as np
import cvxpy as cvx
import sklearn as sk
import scipy.special as scs
import warnings
from sklearn import preprocessing
from sklearn.utils import check_X_y, check_array
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

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


class DWLS(CMRC):
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

    D : `int`, default = None
        Hyperparameter that balances the trade-off between error in
        expectation estimates and confidence of the classification.


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

    weights_alpha : `array`, default = `None`
        The weights alpha(y) associated to each label.

    weights_beta : `array`, default = `None`
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
                 D=None,
                 sigma_=None,
                 hyper_lambda = None,
                 lambda_kernel=0.001,
                 lambda_reg = 1,
                 solver='adam',
                 beta_method='KMM',
                 alpha=0.01,
                 stepsize='decay',
                 mini_batch_size=None,
                 max_iters=None,
                 weights_beta=None,
                 weights_alpha=None,
                 phi='linear',
                 **phi_kwargs):
        self.D = D
        self.beta_ = weights_beta
        self.alpha_ = weights_alpha
        self.lambda_kernel = lambda_kernel
        self.beta_method = beta_method
        self.lambda_reg = lambda_reg
        self.sigma_ = sigma_
        self.hyper_lambda = hyper_lambda
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
        
        if self.alpha_ is None and self.beta_ is None:

            # Compute weights alpha and beta
            # print("Computing both weights")
            if self.beta_method == 'RLLS':
                xWeights, xValid, yWeights, yValid = train_test_split(xTr, yTr, test_size=0.5, random_state=42)
                self.RLLS(xWeights, yWeights, xValid, yValid, xTe, n_classes)
            elif self.beta_method == 'KMM':
                self.LSKMM(xTr, yTr, xTe, n_classes)
            else:
                raise ValueError('Not implemented!')

            if self.D is None:
                Ds = 1 / (1-np.arange(0, 1, 0.1))**2
                Cs = np.max(self.pte_ptr) / np.sqrt(Ds)
                n_Cs = Cs.shape[0]
            else:
                Ds = self.D   
                Cs = np.max(self.pte_ptr) / np.sqrt(Ds)
                n_Cs = 1
                Cs = np.array([Cs])            


            self_aux = self
            min_upper_bound = np.inf
            for i in range(n_Cs):
                self.alpha_ = np.minimum(Cs[i] * 1.0 / self.pte_ptr, np.ones((n_classes,1)))
                self.beta_ = np.minimum(self.pte_ptr, Cs[i] * np.ones((n_classes,1)))            

                # Compute the expectation estimates
                tau_ = self.compute_tau(xTr, yTr)
                if self.hyper_lambda is None:
                    lambda_ = self.compute_lambda(xTe, tau_, n_classes)
                else:
                    lambda_ = self.hyper_lambda * np.ones_like(tau_)

                # Fit the MRC classifier
                self.gamma_ = np.concatenate(( self.beta_[yTr.astype(int), 0] / self.alpha_[yTr.astype(int), 0], np.ones(xTe.shape[0])))
                self.minimax_risk(np.vstack((xTr, xTe)), tau_, lambda_, n_classes)
                if min_upper_bound < self.upper_:
                    min_upper_bound = self.upper_
                    self_aux = self

            self = self_aux

        elif self.alpha_ is None and self.beta_ is not None:
            self.alpha_ = np.ones((n_classes,1))
            self.beta_ = np.reshape(self.beta_, (n_classes, 1))

            tau_ = self.compute_tau(xTr, yTr)
            lambda_ = self.compute_lambda(xTe, tau_, n_classes)
            lambda_ = 0 * lambda_

            # Fit the MRC classifier
            self.gamma_ = self.beta_[yTr, 0] / self.alpha_[yTr, 0]
            self.minimax_risk(xTr, tau_, lambda_, n_classes)

        else:
            # Make sure the size of alpha and beta are as desired
            # print("Weights were given")
            self.beta_ = np.reshape(self.beta_, (n_classes, 1))
            self.alpha_ = np.reshape(self.alpha_, (n_classes, 1))

            tau_ = self.compute_tau(xTr, yTr)
            lambda_ = self.compute_lambda(xTe, tau_, n_classes)

            # Fit the MRC classifier
            self.gamma_ = np.concatenate(( self.beta_[yTr.astype(int), 0] / self.alpha_[yTr.astype(int), 0], np.ones(xTe.shape[0])))
            self.minimax_risk(np.vstack((xTr, xTe)), tau_, lambda_, n_classes)
    
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

        if self.sigma_ is None:
            if n <= 200:
                sigma_ = 0.8 * np.sqrt(d)
            elif n > 200 and n <= 1200:
                sigma_ = 0.5 * np.sqrt(d)
            elif n > 1200:
                sigma_ = 0.3 * np.sqrt(d)
        else:
            sigma_=self.sigma_

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

        M2 = Ky @ np.linalg.inv(Ky + lambda_kernel * np.eye(n))
        A = M2 @ Kx @ M2.T
        M = np.ones((1, t)) @ Kc @ M2.T

        pte_ptr = cvx.Variable((n_classes, 1))
        objective = cvx.Minimize(0.5 * cvx.quad_form(pte_ptr, R.T @ A @ R) - (n / t) * M @ R @ pte_ptr)
        constraints = [
            np.ones((1, n)) @ R @ pte_ptr - n <= n * epsilon,
            n - np.ones((1, n)) @ R @ pte_ptr <= n * epsilon,
            R @ pte_ptr >= 0,
            R @ pte_ptr <= B
        ]

        problem = cvx.Problem(objective, constraints)
        try:
            problem.solve(solver = 'MOSEK', verbose = False)
        except cvx.error.SolverError:
            try:
                problem.solve(solver = 'ECOS', feastol = 1e-4, reltol = 1e-3, abstol = 1e-4)
            except cvx.error.SolverError:
                try:
                    problem.solve(solver = 'SCS')
                except cvx.error.SolverError:
                    raise ValueError('CVXpy couldn\'t find a solution for ' + \
                                     'lambda .... ' + \
                         'The problem is ', problem.status)

        self.pte_ptr = pte_ptr.value
        self.min_KMM = problem.value

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
            problem.solve(solver = 'MOSEK', verbose = False)
        except cvx.error.SolverError:
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