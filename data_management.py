import pickle
import numpy as np
from sklearn.model_selection import train_test_split


class DataManager:
    def __init__(self, dataset_path, test_size=0.3):
        '''
        dataset path: relative path to the data.csv file from the winsconsin breast cancer dataset.
        test_size: value between ]0,1[ representing the fraction of points to be used for the test set.
        '''
        self.dataset_path = dataset_path
        self.test_size = test_size

    def parse_pickle(self, normalize=False):
        '''
        Fonction that parses the data.csv file
        in: path to pickle file
        out: list of dicts containing the data
        '''
        P, U = pickle.load(open(self.dataset_path, 'rb'))
        P_names = [p[0] for p in P]
        U_names = [u[0] for u in U]
        P = np.array([p[-1] for p in P])
        U = np.array([u[-1] for u in U])
        return P, U, P_names, U_names

    def split_training_test(self, X, T, normalize=False):
        x_train, x_test, t_train, t_test = train_test_split(X, T, test_size=self.test_size)
        return x_train, t_train, x_test, t_test

    def normalize(self, X):
        '''
        Function to scale feature between -1 and 1.
        input: feature matrix (2d np array)
        output: scaled feature matrix (2d np array)
        '''
        n = (X-X.min(axis=0))*2
        d = X.max(axis=0) - X.min(axis=0)
        d[d==0] = 1
        return -1 + n/d
