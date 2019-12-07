from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np

class Model:
    def __init__(self, model_type, alpha=1e-4, gamma='scale'):
        '''
        instantiate an ML model (SVM_rbf, SVM_sigmoid, MLP, ***, ***)
        '''
        self.model = model_type
        self.alpha = alpha
        self.gamma = gamma

    def train(self, x_train, t_train):
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
            mlp = MLPClassifier(
                    hidden_layer_sizes=(self.n_layers, *self.n_neuron_per_layer),
                    )
            self.trained_model = mlp.fit(x_train, t_train)

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
        return (predicted_t - real_t)**2

    def hyperpar_tuning():
        '''

        '''
        def cross_val():
            '''

            '''
            pass
        pass

    def bagging():
        pass
