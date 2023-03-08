
# Interpreting Deep Neural Networks with the Package `innsight`

This repository can be used to reproduce the results and figures from the 
paper *"Interpreting Deep Neural Networks with the package innsight"* submitted 
for the Journal of Statistical Software (JSS). It is structured as follows:

* Folder `4_Illustration/` covers the example with the penguin dataset 
(only numerical input variables) and  the melanoma dataset (images and tabular 
inputs).

* Folder `5_Validation/` includes the simulation study of the implemented 
feature attribution methods regarding the mean absolute error considering the 
reference implementations **Captum**, **Zennit**, **iNNvestigate**, and 
**DeepLift**.

* The `Appendix_B/` folder contains the code to reproduce the differences 
between **innsight** and **iNNvestigate** explained in Appendix A for the 
LRP $\alpha$-$\beta$-rule when a bias vector occurs in the model.

Since each reference implementation has different constraints on the 
provided deep learning library and the available packages, the computations 
occur in separated conda environments with the required packages and package 
versions. These conda environments can be created using the R script
`utils/preprocess/create_condaenvs.R` and are essential for reproducing the 
results.

