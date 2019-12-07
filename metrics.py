import numpy as np


class Metrics :
    def __init__(self):
        pass

    def accuracy(self, t_pred, t_test):
        return sum(t_pred==t_test)/len(t_pred)


    def ROC_curve():
        pass

    def recall(self, t_pred, t_test):
        return sum(t_pred[t_pred==t_test]==1.)/sum(t_test==1.)

