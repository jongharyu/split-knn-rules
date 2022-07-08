# One-Nearest-Neighbor Search is All You Need for Minimax Optimal Regression and Classification

This is an accompanying code for the paper "One-Nearest-Neighbor Search is All You Need for Minimax Optimal Regression and Classification" ([arXiv:2202.02464](https://arxiv.org/abs/2202.02464)).

## Requirements
The experiments were run under the following environment:
```text
python==3.8.1
matplotlib==3.3.4
numpy==1.20.0
pandas==1.2.1
py-cpuinfo==7.0.0
scikit-learn==0.24.1
scipy==1.6.0
```

## Datasets
- To run each experiment with a real-world dataset, find the link from Table 2 of the paper and donwload the dataset under `data/dataset-name/`.

## To replicate
- For the synthetic experiment (Section 5.1): run
```commandline
python main_synthetic.py
```

- For the real-world dataset experiment (Section 5.2): run, e.g., 
```commandline
python main.py --parallel True --test-size 0.05 --n-folds 10 --n-trials 10 --algorithm kd_tree --dataset SUSY
```

## To be implemented
- Support node-level parallel computation.

## Acknowledgments
- A cross validation code snippet was adapted from that of this [repository](https://github.com/lirongx/SubNN). 