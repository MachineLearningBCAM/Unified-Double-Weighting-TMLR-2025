# Double-Weighting Multi-Source Covariate Shift
Made with Matlab

This repository is the official implementation of Double-Weighting Adaptation for Multi-Source Covariate Shift. 

The algorithm proposed in the paper provides efficient learning for the proposed Double-Weighting for Multi-Source Covariate Shift (DW-MSCS). We first compute the set of weights $\alpha_s$ and $\beta_s$, for $s\in[S]$, by solving the Multi-Source Kernel Mean Matching (MS-KMM). Then, we learn the classifier's parameters by solving the adaptation of the Minimax Risk Classifiers for Multi-Source Covariate Shift.

## Matlab Code

The folders contains Matlab scripts required to execute the method:

* Experiments_Synthetic folder contain the file to replicate the synthetic experiments in Section 6.2.1.

* Experiments_20News folder contain the files to replicate the experiments of the Table 2 using "20 Newsgroups" dataset.
	* Experiments_20News.m is the main file. It perform the experiments corresponding to Table 2, computing the classification error for ERM LR, KMM, DW-GCS, 2SW-MDA, MS-DRL, CW KMM and DW-MSCS for the 20 Newsgroups dataset. The input to the function (idx1) is a numerical value from 1 to 6 representing the binary classification task to be performed.
	* Experiments_20News_multiclass.m is the other main file. It perform the experiments corresponding to Table 2, computing the classification error for ERM LR, KMM, DW-GCS, 2SW-MDA, MS-DRL, CW KMM and DW-MSCS for the 20 Newsgroups dataset. The input to the function (idx1) is a numerical value from 1 to 4 representing the multiclass classification task to be performed.
  	* generate_sources.m contains the code to generate the sources for the binary experiments as described in the Table 5 of the paper.
	* generate_sources_multiclass.m contains the code to generate the sources for the multiclass experiments as described in the Table 5 of the paper.
  	* generate_train_test.m contains the code to generate the training and testing partition in each repetition.
  	* 20Newsgroups_processed.mat contains the 20 Newsgroups dataset.

* Experiments_Spam folder contain the files to replicate the experiments of the Table 2 using "Spam detection" dataset.
	* Experiments_Spam.m is the main file. It perform the experiments corresponding to Table 2, computing the classification error for ERM LR, KMM, DW-GCS, 2SW-MDA, MS-DRL, CW KMM and DW-MSCS for the Spam dataset. The input to the function are idx1, which is a numerical value from 1 to 6 representing the different number of features selected.
    	* spam_detection.mat contains the spam detection dataset.

* Experiments_Sentiment folder contain the files to replicate the experiments of the Table 2 using "Sentiment" dataset.
  * Experiments_Sentiment_All_Domains.m and Experiments_Sentiment_Three_Domains.m are the main files. They perform the experiments corresponding to Table 2, computing the classification error for ERM LR, KMM, DW-GCS, 2SW-MDA, MS-DRL, CW KMM and MS-DW for the Sentiment dataset. The input to the function Experiments_Sentiment_Three_Domains is idx1, which is a numerical value from 1 to 4 representing the domain that is not going to appear as a training source.
  * domains.mat contains the data of the four domains.
  
* Experiments_DomainNet folder contain the files to replicate the experiments of the Table 2 using "DomainNet" dataset.
	* Experiments_DomainNet.m is the main file. It perform the experiments corresponding to Table 2, computing the classification error for ERM LR, KMM, DW-GCS, 2SW-MDA, MS-DRL, CW KMM and DW-MSCS for the Spam dataset. The input to the function is idx1, which is a numerical value from 1 to 6 representing the six different multiclass problems.
    	* DomainNet_double.mat contains the DomainNet dataset processed for the experiments using a pretrained ResNet-18.
	
* Experiments_Office31 folder contain the files to replicate the experiments of the Table 2 using "Office-31" dataset.
	* Experiments_Office31.m is the main file. It perform the experiments corresponding to Table 2, computing the classification error for ERM LR, KMM, DW-GCS, 2SW-MDA, MS-DRL, CW KMM and DW-MSCS for the Spam dataset. The input to the function is idx1, which is a numerical value from 1 to 4 representing the four different multiclass problems.
    	* DomainNet_double.mat contains the DomainNet dataset processed for the experiments using a pretrained ResNet-50.

* Functions_Library_MSDW folder contains the implementation of the existing methods (ERM LR, KMM, DW-GCS, 2SW-MDA, MS-DRL, and CW KMM) and the proposed approach (DW-MSCS).
  * LR folder contains the functions necessary to implement the ERM LR approach.
  * KMM folder contains the functions necessary to implement the KMM approach.
  * DWGCS folder contains the functions necessary to implement the DW-GCS approach.
  * 2SWMDA folder contains the functions necessary to implement the 2SW-MDA approach.
  * MSDRL folder contains the functions necessary to implement the MS-DRL approach.
  * CWKMM folder contains the functions necessary to implement the CW KMM approach.
  
  * DWMSCS folder contains the functions necessary to implement the proposed DW-MSCS approach.
    * MSKMM.m computes the estimated sets of weights $\beta_{s}$ and $\alpha_{s}$, for $s=1,2,\ldots,S$ solving the multi-source kernel mean matching.
    * MSDW_parameters.m obtains the sets of mean vector estimates $\tau_s$ and confidence vectors $\lambda_s$, for $s=1,2,\ldots,S$.
    * MSDW_learning.m solves the convex MRC optimization problem using double-weighting and obtains the classifier parameters for multi-source covariate shift.
    * MSDW_prediction.m assigns labels to instances and returns the classification error.
    
Note that cvx package for Matlab is needed, as well as a mosek licence in order to solve the multiple optimization problems. In case you want to use another cvx solver, just change the line "cvx_ solver mosek" to "cvx_solver <solver>".

Link to download cvx package: https://cvxr.com/cvx/download/
Link to dowload mosek licence: https://www.mosek.com/products/academic-licenses/