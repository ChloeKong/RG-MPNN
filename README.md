# Integrating concept of pharmacophore with Graph Neural Networks for chemical property prediction and interpretation

This repository contains the source code and the data.

## RG-MPNN

![](toc.png)



## Setup and dependencies 

Dependencies:
- python 3.7
- pytorch = 1.7.1
- torch-cluster = 1.5.9
- torch-geometric = 1.7.2
- torch-scatter = 2.0.7
- torch-sparse = 0.6.9
- torch-spline-conv = 1.2.1
- RDkit = 2021.03.3
- numpy
- pandas

## Data sets

The data sets are provided as .csv files in a directory called 'data', including benchmark datasets and kinase datasets used in this work. 


---

## Using

1.`MyNet_Classification` generates input, train and test classification models. For example,

```
python MyNet_Classification.py \
    --epochs 100 \
    --dataset BACE \
    --model RGNN 
```


2.`MyNet_Regression` generates input, train and test regression models. For example,
```
python MyNet_Regression.py \
    --epochs 100 \
    --dataset Lipo \
    --model RGNN
```

---

## Author

Yue Kong 

Aixia Yan

## Citation

