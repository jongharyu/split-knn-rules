# Minimax-Optimal Fixed-k-Nearest Neighbor Algorithms

This is an accompanying code for the paper "Minimax Algorithms with Fixed-$k$-Nearest Neighbors" ([arXiv:2202.02464v2](https://arxiv.org/abs/2202.02464)).

## Datasets
- To run each experiment with a real-world dataset for classification and regression, find the link from Table 2 of the paper and download the dataset under `data/dataset-name/`.

## To replicate

### Classification and Regression
- For the synthetic data: run
```commandline
python main_synthetic.py
```

- For the real-world dataset experiment: run, e.g., 
```commandline
python main.py --parallel True --test-size 0.05 --n-folds 10 --n-trials 10 --algorithm kd_tree --dataset SUSY
```

## To be implemented
- Support node-level parallel computation.

## Acknowledgments
- A cross validation code snippet was adapted from that of this [repository](https://github.com/lirongx/SubNN).