from data_management import *
from models import *
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Predict wether a breast cancer tumor is malign or benin based on biopsy data.')
parser.add_argument('data_file', type=str, help='relative path to the csv data file.')
args = parser.parse_args()


if __name__ == "__main__":
    fpath = args.data_file
    data_manager = DataManager(fpath)
    x_train, t_train, x_test, t_test = data_manager.get_data()
    print(x_train.shape, t_train.shape)
