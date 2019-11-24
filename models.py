from sklearn import svm, neural_network
import numpy as np

class Model:
    def __int__(self, model_type, gamma='scale'):
        '''
        instantiate an ML model (SVM_rbf, SVM_sigmoid, MLP, ***, ***)
        '''
        self.model = model_type
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
        # mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
        #                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
        if self.model == 'MLP':
            mlp = neural_network.MLPClassifier(
                    hidden_layer_sizes=(self.n_layers,),
                    )

    def predict(self, x):
        pass

    def hyperpar_tuning():
        '''

        '''
        def cross_val():
            '''

            '''
        pass

    def bagging():
        pass
