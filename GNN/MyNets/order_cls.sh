#! /bin/bash

for dataset in BACE BBBP HIV MUV Tox21 ToxCast SIDER ClinTox
do
    for i in 88 100 121 888 666
    do
        for model in RGNN NN AFP GNN ReduceNN ReduceAFP
        do
            python MyNet_Classification.py --dataset ${dataset} --model ${model} --cuda 3 --seed ${i} --dropout 0.3 --choose_model 'AUC'
        done
    done
done


