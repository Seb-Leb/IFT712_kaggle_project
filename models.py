from sklearn import svm, neural_network
import numpy as np
from sklearn import svm, neural_network

class Model:
    def __int__(self, ):
        pass

    def svm(self, kernel):
        pass

    def neural_net(self, n_layers, n_neurons):
        '''
        instantiate an ML model (SVM_rbf, SVM_sigmoid, MLP, ***, ***)
        '''
        pass

    def train(self, x_train, t_train):
        # SVM RBF kernel
        clf = svm.SVC(gamma=self.gamma)
        self.trained_model = clf.fit(x_train, t_train)

        # SVM sigmoid kernel

        # Multi layer perceptron

    def predict(self, x):
        pass

    def hyperpar_tuning():
        '''

        '''
        def cross_val():
            '''

            '''
        pass
    pass
