import itertools as itt
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np

class Model:
    def __init__(self, model_type, niter=1000, C=1., M=20, alpha=1e-4, gamma='scale', neuron_per_layer=(20,20)):
        '''
        instantiate an ML model (SVM_rbf, SVM_sigmoid, MLP, ***, ***)
        '''
        self.model = model_type
        self.M     = M
        self.niter = niter
        self.C     = C
        self.alpha = alpha
        self.gamma = gamma
        self.neuron_per_layer = neuron_per_layer

    def train(self, x_train, t_train):
        # Linear Classifier
        if self.model == 'linear':
            clf = svm.LinearSVC(penalty='l2', C=self.C, max_iter=self.niter)
            self.trained_model = clf.fit(x_train, t_train)

        # SVM RBF kernel
        if self.model == 'svm_rbf':
            clf = svm.SVC(gamma=self.gamma, kernel='rbf')
            self.trained_model = clf.fit(x_train, t_train)

        # SVM sigmoid kernel
        if self.model == 'svm_sigmoid':
            clf = svm.SVC(gamma=self.gamma, kernel='sigmoid')
            self.trained_model = clf.fit(x_train, t_train)

        # Multi layer perceptron
        if self.model == 'MLP':
            clf = MLPClassifier(
                    hidden_layer_sizes=(self.neuron_per_layer),
                    )
            self.trained_model = clf.fit(x_train, t_train)

    def predict(self, x):
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

    def error(self, predicted_t, real_t):
        '''
        Error calculation on the positive trainning labels.
        '''
        return (predicted_t - real_t)**2

    def hyperpar_tuning(self, x_train, t_train, hyperpar_ranges, search_method='grid'):
        '''
        Hyper parameter search with k-fold cross validation.
        '''

        hyperpar_sets = itt.product()
        hyperpar_tests = dict()
        for hyperpar_set in hyperpar_sets:
            score = cross_val(x_val, t_val, hyperpar_set)
            hyperpar_tests[score] = hyperpar_set

        best_hyperpars = hyperpar_tests[max(hyperpar_tests.keys())]

        return best_hyperpars

    def kfold_cross_val(self, x_train, t_train, k=5):
        '''

        '''
        skf  = StratifiedKFold(n_splits=k)
        errs = []
        for train_idx, test_idx in skf.split(x_train, t_train):
            x_tr, t_tr = x_train[train_idx], t_train[train_idx]
            x_ts, t_ts = x_train[test_idx], t_train[test_idx]
            self.train(x_tr, t_tr)
            t_pred = self.predict(x_ts, t_ts)
            errs.append(self.error(t_s, t_pred))
        return np.mean(errs)


class Ensemble:
    def __init__(self):
        pass

    def bagging():
        pass
