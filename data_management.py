import numpy


class data_manager:
    def __init__(self, dataset_path, train_test_split_ratio):
        self.dataset_path = dataset_path
        self.train_test_split_ratio = train_test_split_ratio

    def parse_csv(file_path):
        '''
        Fonction that parses the data.csv file
        in: path to csv file
        out: list of dicts containing the data
        '''
        with open(file_path, 'r') as f:
            for n,l in enumerate(f):
                ls = l.strip().split(',')
                if n==0:
                    keys = ls
                    continue
                line = dict(zip(keys, ls))
                for k in line.keys():
                    if k not in ('id', 'diagnosis'):
                        line[k] = float(line[k])
                data.append(line)
        return data

    def split_traning_test(self):
        return x_train, t_train, x_test, t_test

    def get_data(self, ):
