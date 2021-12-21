#! /bin/bash

for dataset in AKT1 AURKA BRAF BTK CDK2 CK1 EGFR MAP4K2 mTOR PIM1
do
    for i in 88 100 121 888 666
    do
        for model in RGNN NN AFP GNN ReduceNN ReduceAFP
        do
            python MyNet_Classification.py --dataset ${dataset} --model ${model}  --cuda 0 --dropout 0.3 --seed ${i} --choose_model 'AUC'
        done
    done
done

