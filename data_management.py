import numpy

def parse_csv(file_path):
    '''
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

def split_traning_test():
    return x_train, t_train, x_test, t_test

