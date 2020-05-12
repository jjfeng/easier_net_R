# EASIER-net for R

Feng, Jean, and Noah Simon. 2020. “Ensembled Sparse-Input Hierarchical Networks for High-Dimensional Datasets.” arXiv [stat.ML]. arXiv. http://arxiv.org/abs/2005.04834.

This repository contains the R code for fitting EASIER-net.

The python repository that fits EASIER-net and reproduces paper results is at https://github.com/jjfeng/easier_net.

## Installation
Install the package from github: `devtools::install_github("jjfeng/easier_net_R", build_vignettes = TRUE, force=TRUE)`
Load the library via `library(easiernet)`.

## Quick-start

See vignettes: `browseVignettes('easiernet')`
`Classification` provides an example EASIER-net procedure for classification.
`Regression` provides an example EASIER-net procedure for regression.
`Cross-validation` provides an example EASIER-net procedure for tuning penalty parameters via cross-validation.
