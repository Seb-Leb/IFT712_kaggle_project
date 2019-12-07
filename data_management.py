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

    def parse_pickle(self):
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

    def split_training_test(self, P, U, test_size=0.3, PU_ratio=0.5, normalize=False):
        if normalize:
            len_p  = len(P)
            X_norm = self.normalize(np.concatenate((P, U), axis=0))
            P = X_norm[:len_p]
            U = X_norm[len_p:]
        p_train, p_test, _, _ = train_test_split(P, np.array([1.,]*len(P)), test_size=test_size)
        nU = int(len(p_train)*(1+PU_ratio))
        u_train, u_test, _, _ = train_test_split(U, np.array([0.,]*len(U)), train_size=nU)

        x_train = np.concatenate((p_train, u_train), axis=0)
        x_test  = np.concatenate((p_test, u_test), axis=0)
        t_train = np.concatenate((np.array([1.,]*len(p_train)), np.array([0.,]*len(u_train))), axis=0)
        t_test  = np.concatenate((np.array([1.,]*len(p_test )), np.array([0.,]*len(u_test ))), axis=0)

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
