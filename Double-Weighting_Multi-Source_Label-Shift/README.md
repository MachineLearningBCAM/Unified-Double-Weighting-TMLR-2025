# Double-Weighting Multi-Source Label Shift
Made with Matlab

This repository is the official implementation of Double-Weighting Adaptation for Multi-Source Label Shift. 

The algorithm proposed in the paper provides efficient learning for the proposed Double-Weighting for Multi-Source Label Shift (DW-MSLS). We first compute the set of weights $\alpha_s$ and $\beta_s$, for $s\in[S]$ and then, we learn the classifier's parameters by solving the adaptation of the Minimax Risk Classifiers for Multi-Source Covariate Shift.

## Matlab Code

The folders contains Matlab scripts required to execute the method:

* Experiments_MSLS folder contain the files to replicate the experiments of the Table 3 using "Sentiment" dataset.
  * Experiments_MSLS.m is the main file. It performs the experiments corresponding to Table 3, computing the classification error for LR, KMM, DW-LS, LWC KMM, and DW-MSLS for the Sentiment dataset. The input to the function Experiments_MSLS are idx1, which is a numerical value from 1 to 4 representing the domain that is not going to appear as a training source, and alpha_type, which is a numerical value from 1 to 5 that determines the parameter $\gamma$ of the dirichlet shift.
  * domains.mat contains the data of the four domains.

* Functions_Library_DW-MSLS folder contains the implementation of the existing methods (LR, KMM, DW-LS, and LWC KMM) and the proposed approach (DW-MSLS).
  * LR folder contains the functions necessary to implement the LR approach.
  * Rw folder contains the functions necessary to implement the KMM and LWC KMMapproaches.
  * DWMSLS folder contains the functions necessary to implement the DW-LS and DW-MSLS approaches.
  
  * DWMSLS folder contains the functions necessary to implement the proposed DW-MSCS approach.
    * MSLS_KMM.m computes the estimated sets of weights $\beta_{s}$ and $\alpha_{s}$, for $s=1,2,\ldots,S$.
    * MSLS_parameters.m obtains the sets of mean vector estimates $\tau_s$ and confidence vectors $\lambda_s$, for $s=1,2,\ldots,S$ and solves the convex MRC optimization problem using double-weighting and obtains the classifier parameters for multi-source label shift.
    * MSLS_prediction.m assigns labels to instances and returns the classification error.
    
Note that cvx package for Matlab is needed, as well as a mosek licence in order to solve the multiple optimization problems. In case you want to use another cvx solver, just change the line "cvx_solver mosek" to "cvx_solver <solver>".

Link to download cvx package: https://cvxr.com/cvx/download/
Link to dowload mosek licence: https://www.mosek.com/products/academic-licenses/