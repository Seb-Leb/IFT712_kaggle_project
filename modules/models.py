#####
# Jeremie Beliveau-Lefebvre (04494470)
# Sebastien Leblanc         (18206273)
###

import itertools as itt
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import numpy as np
import pickle
from modules.metrics import *

class Model:
    def __init__(self, model_type, max_iter=1000, **kwargs):
        '''
        instantiate an ML model (SVM_rbf, SVM_sigmoid, MLP, ***, ***)
        '''
        self.max_iter = max_iter

        self.model = model_type
        if self.model == 'linear':
            hyperpars       = {'C':1., 'M':5}
            hyperpar_ranges = {
                    'C':range(1,6),
                    'M':range(2,6),
                    }
        elif self.model == 'SVM':
            hyperpars       = {'C':1., 'alpha':1e-4, 'gamma':'scale', 'kernel':'rbf'}
            hyperpar_ranges = {
                    'C':range(1,7),
                    'gamma':1/np.logspace(np.log10(1e-9), np.log10(2), 7),
                    }
            if 'kernel' in kwargs:
                hyperpars['kernel'] = kwargs['kernel']
        elif self.model == 'MLP':
            hyperpars       = {'n_neurons':20, 'n_layers':2, 'solver':'sgd'}
            hyperpar_ranges = {
                    'n_neurons' : range(20,50,5),
                    'n_layers'  : range(1,5),
                    }

        for hpp in hyperpars:
            if hpp not in kwargs:
                setattr(self, hpp, hyperpars[hpp])
            else:
                setattr(self, hpp, kwargs[hpp])

        self.hyperpar_ranges = hyperpar_ranges

    def train(self, x_train, t_train, return_model=False):
        # Linear Classifier
        if self.model == 'linear':
            x_hd = self.polynomial_basis_fun(x_train)
            clf = svm.LinearSVC(penalty='l2', C=self.C, max_iter=self.max_iter)
            self.trained_model = clf.fit(x_hd, t_train)

        # SVM
        if self.model == 'SVM':
            clf = svm.SVC(C=self.C, gamma=self.gamma, kernel=self.kernel, probability=True)
            self.trained_model = clf.fit(x_train, t_train)

        # Multi layer perceptron
        if self.model == 'MLP':
            neuron_per_layer = (self.n_neurons,)*self.n_layers
            clf = MLPClassifier(
                    hidden_layer_sizes=neuron_per_layer,
                    max_iter=self.max_iter,
                    )
            self.trained_model = clf.fit(x_train, t_train)
        if return_model:
            return self

    def predict(self, x, proba=False):
        if self.model == 'linear':
            x = self.polynomial_basis_fun(x)
        if proba:
            return self.trained_model.predict_proba(x)[:,1]
        return self.trained_model.predict(x)

    def polynomial_basis_fun(self, X):
        '''
        prodect data into a higher dimentional space.
        :input: X ndarray
        '''
        phi_X = X
        for m in range(1, self.M):
            phi_X = np.concatenate((phi_X, X**m), axis=1)
        return phi_X


    def hyperpar_tuning(self, x_train, t_train):
        '''
        Hyper parameter search with k-fold cross validation.
        '''

        hyperpar_sets = [dict(zip(self.hyperpar_ranges.keys(), v)) for v in itt.product(*self.hyperpar_ranges.values())]

        hyperpar_results = dict()
        for hyperpar_set in hyperpar_sets:
            for hpp in hyperpar_set:
                setattr(self, hpp, hyperpar_set[hpp])
            score = self.kfold_cross_val(x_train, t_train)
            hyperpar_results[score] = hyperpar_set

        best_hyperpars = hyperpar_results[max(hyperpar_results.keys())]

        return best_hyperpars

    def kfold_cross_val(self, x_train, t_train, k=3, return_array=False):
        '''

        '''
        metric = Metrics()
        skf  = StratifiedKFold(n_splits=k)
        scores = []
        for train_idx, test_idx in skf.split(x_train, t_train):
            x_tr, t_tr = x_train[train_idx], t_train[train_idx]
            x_ts, t_ts = x_train[test_idx], t_train[test_idx]
            self.train(x_tr, t_tr)
            t_pred = self.predict(x_ts)
            scores.append(metric.LL_score(t_ts, t_pred))
        if return_array:
            return scores
        return np.mean(scores)

    def save_model(self,):
        if self.traine_model is not None:
            pickle.dump(
                    self.trained_model,
                    open('saved_models/{}_{}.pkl'.format(self.model, str(datetime.now())), 'wb'))


class Ensemble:
    def __init__(self, model, hyperpars=None, M=10, **kwargs):
        self.model = Model(model)
        if hyperpars is not None:
            for hpp in hyperpars:
                setattr(self.model, hpp, hyperpars[hpp])
        self.M = M
        self.model_array = []

    def bagging_train(self, x, t):
        '''
        Bootstrap data samples and train an array of models.
        '''
        self.model_array = []
        n_samp = int(len(x)/self.M)*3
        print('using {} samples per model.'.format(n_samp))
        for i in range(self.M):
            x, t = resample(x, t, n_samples=n_samp)
            self.model_array.append(
                    self.model.train(x, t, return_model=True)
                    )

    def bagging_cross_val(self, x_train, t_train, k=10):
        metric = Metrics()
        skf  = StratifiedKFold(n_splits=k)
        scores = []
        for train_idx, test_idx in skf.split(x_train, t_train):
            x_tr, t_tr = x_train[train_idx], t_train[train_idx]
            x_ts, t_ts = x_train[test_idx], t_train[test_idx]
            self.bagging_train(x_tr, t_tr)
            t_pred = self.predict(x_ts)
            scores.append(metric.LL_score(t_ts, t_pred))
        return scores


    def predict(self, x, proba=False):
        T_pred = np.zeros(len(x))
        T_pred_bag = np.zeros(len(x))
        for m in self.model_array:
            if proba:
                T_pred_bag += m.predict(x, proba=True)
                continue
            T_pred_bag += m.predict(x)
        T_pred_bag /= len(self.model_array)
        if proba:
            return T_pred_bag
        T_pred[T_pred_bag>0.5] = 1
        return T_pred
