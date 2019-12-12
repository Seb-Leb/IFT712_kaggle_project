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


    def rc_ll_curve(self, t_pred_proba, t_test, iterations=50):
        '''

        '''
        
        if iterations<10:
            iterations = 10
        
        rc, ll = [], []
        for threshold in np.linspace(0., 1., iterations):
            print("Itr")
            tp   = np.zeros(len(t_pred_proba))
            tp[t_pred_proba > threshold] = 1.
            rc.append(self.recall(tp, t_test))
            ll.append(self.LL_score(tp, t_test))
        return rc, ll

