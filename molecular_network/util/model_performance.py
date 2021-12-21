from sklearn import metrics
from sklearn.metrics import (accuracy_score,matthews_corrcoef,
                            r2_score,mean_absolute_error,confusion_matrix,
                            mean_squared_error,roc_auc_score)

import pandas as pd
import numpy as np


class modelEvaluator():
    def __init__(self,y_true,y_pred,y_prob=None,model_kind=None):


        self.model_kind = model_kind
        self.y_prob = y_prob

        if model_kind == None:
            if len(np.unique(y_true))<=5: 
                self.__Cls_metrics(y_true,y_pred,y_prob)
                self.model_kind = 'cls'
            else:
                self.__Rgr_metrics(y_true,y_pred)
                self.model_kind = 'rgs'
        elif model_kind == 'cls':
            self.__Cls_metrics(y_true,y_pred,y_prob)

        else:
            self.__Rgr_metrics(y_true,y_pred)
            
    def __Cls_metrics(self,y_true,y_pred, y_prob):
 
        y_true=np.array(y_true)
        y_pred=np.array(y_pred)
        self.accuracy = round(accuracy_score(y_true, y_pred),3)
        self.mcc = round(matthews_corrcoef(y_true, y_pred),3)
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(y_true,y_pred).ravel()
        self.precision = round(float(self.tp) / float(self.tp + self.fp + 0.0001),3)
        self.recall = round(float(self.tp) / float(self.tp + self.fn),3)
        self.se = round(float(self.tp) / float(self.tp + self.fn),3)
        self.sp = round(float(self.tn) / float(self.tn + self.fp),3) 
        
        if self.y_prob is not None:
            self.auc = round(roc_auc_score(y_true,y_prob),3)
            self.roc = metrics.roc_curve(y_true, y_prob)
        
    def __Rgr_metrics(self,y_true,y_pred):

        y_true=np.array(y_true)
        y_pred=np.array(y_pred)
        self.r2 = round(r2_score(y_true, y_pred),3)
        self.rmse = round(mean_squared_error(y_true, y_pred)**0.5,3)
        self.mae = round(mean_absolute_error(y_true, y_pred),3)

    def get_performance(self):
        print('model kind: %s' %(self.model_kind))
        pfm_dict= dict()

        if self.model_kind == 'cls':
            pfm_dict['accuracy'] = self.accuracy
            pfm_dict['mcc'] = self.mcc
            pfm_dict['precision'] = self.precision
            pfm_dict['recall'] = self.recall
            pfm_dict['se'] = self.se
            pfm_dict['sp'] = self.sp
            
            if self.y_prob is not None:
                pfm_dict['auc'] = self.auc

        if self.model_kind == 'rgs':
            pfm_dict['r2'] = self.r2
            pfm_dict['rmse'] = self.rmse
            pfm_dict['mae'] = self.mae
        
        print('performance dict:', pfm_dict)
        return pfm_dict