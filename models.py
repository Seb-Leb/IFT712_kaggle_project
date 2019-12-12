import itertools as itt
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import numpy as np
from metrics import *

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
                    'C':range(1,3),
                    'M':range(2,4),
                    }
        elif self.model == 'SVM':
            hyperpars       = {'C':1., 'alpha':1e-4, 'gamma':'scale', 'kernel':'rbf'}
            hyperpar_ranges = {
                    'C':range(1,4),
                    'gamma':1/np.logspace(np.log10(1e-9), np.log10(2), 5),
                    }
            if 'kernel' in kwargs:
                hyperpars['kernel'] = kwargs['kernel']
        elif self.model == 'MLP':
            hyperpars       = {'neuron_per_layer':(20,20), 'solver':'sgd'}
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

    def train(self, x_train, t_train):
        # Linear Classifier
        if self.model == 'linear':
            x_hd = self.polynomial_basis_fun(x_train)
            clf = svm.LinearSVC(penalty='l2', C=self.C, max_iter=self.max_iter)
            self.trained_model = clf.fit(x_hd, t_train)

        # SVM
        if self.model == 'SVM':
            clf = svm.SVC(C=self.C, gamma=self.gamma, kernel=self.kernel)
            self.trained_model = clf.fit(x_train, t_train)

        # Multi layer perceptron
        if self.model == 'MLP':
            clf = MLPClassifier(
                    hidden_layer_sizes=self.neuron_per_layer,
                    max_iter=self.max_iter,
                    )
            self.trained_model = clf.fit(x_train, t_train)
        return self.trained_model

    def predict(self, x):
        if self.model == 'linear':
            x = self.polynomial_basis_fun(x)
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


    def hyperpar_tuning(self, x_train, t_train, search_method='grid'):
        '''
        Hyper parameter search with k-fold cross validation.
        '''

        hyperpar_sets = [dict(zip(self.hyperpar_ranges.keys(), v)) for v in itt.product(*self.hyperpar_ranges.values())]

        hyperpar_results = dict()
        for hyperpar_set in hyperpar_sets:
            for hpp in hyperpar_set:
                setattr(self, hpp, hyperpar_set[hpp])
            if self.model == 'MLP':
                self.neuron_per_layer = (self.n_neurons,)*self.n_layers
            score = self.kfold_cross_val(x_train, t_train)
            hyperpar_results[score] = hyperpar_set

        best_hyperpars = hyperpar_results[max(hyperpar_results.keys())]

        return best_hyperpars

    def kfold_cross_val(self, x_train, t_train, k=3):
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
        return np.mean(scores)


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
        for i in range(self.M):
            x, t = resample(x, t, n_samples=2000)
            self.model_array.append(
                    self.model.train(x, t)
                    )

    def predict(self, x):
        T_pred = []
        for m in self.model_array:
            T_pred.append(m.predict(x))
        if sum(T_pred)/len(T_pred) > 0.5:
            return 1.
        return 0.
