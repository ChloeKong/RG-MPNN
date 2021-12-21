#! /bin/bash

for dataset in AURKA AKT1 BRAF BTK CDK2 CK1 EGFR MAP4K2 mTOR PIM1
do
    for i in 18 1021 1002 8803 6604
    do
        for model in RGNN NN AFP GNN ReduceNN ReduceAFP
        do
            python MyNet_Classification.py --dataset ${dataset} --model ${model}  --cuda 2 --dropout 0.3 --choose_model 'AUC' --checkpoints
        done
    done
done




