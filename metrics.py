import numpy as np


class Metrics :
    def __init__(self):
        pass

    def accuracy(self, t_pred, t_test):
        return sum(t_pred==t_test)/len(t_pred)


    def LL_score(self, t_pred, t_test):
        if sum(t_pred==1.)==0.:
            return 0.
        return self.recall(t_pred, t_test)**2/(sum(t_pred==1.)/len(t_pred))


    def recall(self, t_pred, t_test):
        if sum(t_pred==1.)==0 or sum(t_test==1.)==0:
            return 0.
        return sum(t_pred[t_pred==t_test]==1.)/sum(t_test==1.)

