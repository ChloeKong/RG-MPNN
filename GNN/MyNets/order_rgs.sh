#! /bin/bash

for dataset in ESOL FreeSolv Lipo
do
    for model in RGNN NN AFP GNN ReduceNN ReduceAFP
    do
        for i in 100 101 102 103 104
        do
            python MyNet_Regression.py --dataset ${dataset} --model ${model} --cuda 3  --epochs 100 --seed ${i} --dropout 0
            python MyNet_Regression.py --dataset ${dataset} --model ${model} --cuda 3  --epochs 100 --seed ${i} --dropout 0.1
            python MyNet_Regression.py --dataset ${dataset} --model ${model} --cuda 3  --epochs 100 --seed ${i} --dropout 0.2
            python MyNet_Regression.py --dataset ${dataset} --model ${model} --cuda 3  --epochs 100 --seed ${i} --dropout 0.3
            python MyNet_Regression.py --dataset ${dataset} --model ${model} --cuda 3  --epochs 100 --seed ${i} --dropout 0.4
        done
    done
done

