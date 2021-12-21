#!/usr/bin/env python
# coding: utf-8

# 计算metrics脚本

import numpy as np
import torch
from sklearn.metrics import (accuracy_score,matthews_corrcoef,
                            r2_score,mean_squared_error,
                            mean_absolute_error,confusion_matrix, roc_auc_score)


def metrics_C(y_true, y_pred, y_probas, prefix = ""):
    """
    parameter
    -----
    y_true/y_pred: 1D-np.array
    prefix: str, return时返回字典的键前缀
    
    return
    -----
    return: dict
    """
    result = {}

    cm = confusion_matrix(y_true,y_pred)  
    result[prefix+"TN"] = cm[0,0]
    result[prefix+"FP"] = cm[0,1]
    result[prefix+"FN"] = cm[1,0]
    result[prefix+"TP"] = cm[1,1]
    
    result[prefix+"SE"]  = result[prefix+"TP"] / (result[prefix+"TP"] + result[prefix+"FN"])
    result[prefix+"SE"] = round(result[prefix+"SE"],3)
    result[prefix+"SP"]  = result[prefix+"TN"] / (result[prefix+"TN"] + result[prefix+"FP"])
    result[prefix+"SP"] = round(result[prefix+"SP"], 3)
    result[prefix+"ACC"] = accuracy_score(y_true, y_pred)
    result[prefix+"ACC"] = round(result[prefix+"ACC"], 3)
    result[prefix+"MCC"] = matthews_corrcoef(y_true, y_pred)
    result[prefix+"MCC"] = round(result[prefix+"MCC"],3)
    result[prefix+"AUC"] = roc_auc_score(y_true, y_probas)
    result[prefix+"AUC"] = round(result[prefix+"AUC"],3)


    return result


def metrics_R(y_true, y_pred, prefix = ""):
    """
    parameter
    -----
    y_true/y_pred: 1D-np.array   
    prefix: str, return时返回字典的键前缀
    
    return
    -----
    return: dict
    """
    result = {}
    result[prefix+"MSE"] = mean_squared_error(y_true, y_pred)
    result[prefix+"MSE"] = round(result[prefix+"MSE"], 3)
    result[prefix+"RMSE"] = np.sqrt(result[prefix+"MSE"])
    result[prefix+"RMSE"] = round(result[prefix+"RMSE"], 3)
    result[prefix+"MAE"] = mean_absolute_error(y_true, y_pred)
    result[prefix+"MAE"] = round(result[prefix+"MAE"], 3)
    result[prefix+"r2"] = r2_score(y_true, y_pred)
    result[prefix+"r2"] = round(result[prefix+"r2"], 3)
    return result


'''
测试代码
y_true = np.array([1,0,1,1,1])
y_pred = np.array([0,1,0,1,0])

results_C = metrics_C(y_true,y_pred)
print(results_C)

results_R = metrics_R(y_true,y_pred)
print(results_R)
'''

