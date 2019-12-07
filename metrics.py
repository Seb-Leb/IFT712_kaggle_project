import numpy as np


class Metrics :
    def __init__(self, nb_train, nb_test, lineairement_sep):
        self.nb_train = nb_train
        self.nb_test = nb_test
        self.lineairement_sep = lineairement_sep

    def accuracy(t_pred, t_test):
        return sum(t_pred==t_test)/len(t_pred)


    def ROC_curve():
        pass

    def recall(t_pred, t_test):
        return sum(t_pred==t_test and t_pred==1.)/sum(t_test==1.)

