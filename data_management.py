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

    def parse_csv(self):
        '''
        Fonction that parses the data.csv file
        in: path to csv file
        out: list of dicts containing the data
        '''
        data = []
        with open(self.dataset_path, 'r') as f:
            for n,l in enumerate(f):
                ls = l.strip().split(',')
                if n==0:
                    keys = [x.strip('"') for x in ls]
                    continue
                line = dict(zip(keys, ls))
                for k in line.keys():
                    if k not in ('id', 'diagnosis'):
                        line[k] = float(line[k])
                data.append(line)
        X = np.array([[v for k,v in x.items() if k not in {'id', 'diagnosis'}] for x in data])
        T = np.array([1. if x['diagnosis']=='M' else 0. for x in data])
        return X, T

    def split_training_test(self, X, T, norm=False):
        if norm:
            X = self.normalize(X)
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
