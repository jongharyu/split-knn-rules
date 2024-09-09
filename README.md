# Minimax-Optimal Fixed-k-Nearest Neighbor Algorithms

This is an accompanying code for the paper "Minimax Algorithms with Fixed-$k$-Nearest Neighbors" ([arXiv:2202.02464v2](https://arxiv.org/abs/2202.02464)).

## Datasets
- To run each experiment with a real-world dataset for classification and regression, find the link from Table 2 of the paper and download the dataset under `data/dataset-name/`.

## To replicate

### Classification and Regression
- To replicate the mixture of Gaussians experiment, run
```commandline
python main_synthetic.py
```
- For the plots in the paper, check the jupyter notebook `notebooks/exp_cls_mog_results.ipynb` 

- For the real-world dataset experiment: run, e.g., 
```commandline
python main.py --parallel True --test-size 0.05 --n-folds 10 --n-trials 10 --algorithm kd_tree --dataset SUSY
```
- For the validation error profile plot in the paper, check the jupyter notebook `notebooks/exp_cls_real_data_val_errors.ipynb`

### Density Estimation
- To replicate the random mixture of Gaussians experiment, run the jupyter notebook `notebooks/exp_density_mog.ipynb`

## To be implemented
- Support node-level parallel computation.

## Acknowledgments
- A cross validation code snippet was adapted from that of this [repository](https://github.com/lirongx/SubNN).