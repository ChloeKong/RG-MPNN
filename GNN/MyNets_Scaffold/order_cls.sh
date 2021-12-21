#! /bin/bash

for dataset in BACE BBBP Tox21 ToxCast SIDER ClinTox HIV MUV
do
    for i in 100 101 102 103 104
    do
        for model in RGNN NN AFP GNN ReduceNN ReduceAFP
        do
            python MyNet_Classification.py --dataset ${dataset} --model ${model} --cuda 2 --epochs 100  --dropout 0.1
        done
    done
done

