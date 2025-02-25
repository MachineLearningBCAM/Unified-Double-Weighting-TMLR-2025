# Double-Weighting Single-Source Label Shift
Made with Python

The algorithm proposed in the paper provides efficient learning for the proposed Double-Weighting for Single-Source Label Shift (DW-LS). We first compute the set of weights $\alpha$ and $\beta$, and then, we learn the classifier's parameters by solving the adaptation of the Minimax Risk Classifiers for Single-Source Label Shift.

## Python Code

The folders contains Python scripts required to execute the method:

* synthetic.py replicate the synthetic experiments in Section 6.1.1.

* datasets_tweak.py replicate the experiments of the Table 2 when we generate the label shift using tweak-one shift.

* datasets_knock.py replicate the experiments of the Table 2 when we generate the label shift using knock-out shift.

* datasets_dirichlet.py replicate the experiments of the Table 2 when we generate the label shift using dirichlet shift.

* rwls.py contains the class RWLS created to run the experiments for methods based on a reweighted approach.

* dwls.py contains the class DWLS created to run the experiments for methods based on double-weighting approach.
  
Note that for running the experiments, it is needed to install the packages in the requirements.txt file, as well as a mosek licence in order to solve some of the optimization problems efficiently.

Link to dowload mosek licence: https://www.mosek.com/products/academic-licenses/