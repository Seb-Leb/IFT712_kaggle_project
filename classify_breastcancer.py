from data_management import *
from models import *
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Predict wether a breast cancer tumor is malign or benin based on biopsy data.')
parser.add_argument('data_file', type='str', help='relative path to the csv data file.')
args = parser.parse_args()


if __name__ == "__main__":

    args.data_file
