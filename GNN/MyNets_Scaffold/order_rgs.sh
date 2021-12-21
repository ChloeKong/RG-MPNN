#! /bin/bash


for dataset in ESOL FreeSolv Lipo
do
    for model in RGNN NN AFP ReduceAFP ReduceNN GNN
    do
        for i in 100 101 102 103 104
        do
            python MyNet_Regression.py --dataset ${dataset} --model ${model} --cuda 2  --epochs 100
        done
    done
done

