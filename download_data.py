


#####    Kaggle project
# Jeremie Beliveau-Lefebvre (04494470)
# Sebastien Leblanc         (18206273)
###

import numpy as np
from tqdm import tqdm
import kaggle
#import requests


class load :
    def __init__(self, nb_train, nb_test, lineairement_sep):
        self.nb_train = nb_train
        self.nb_test = nb_test
        self.lineairement_sep = lineairement_sep

    def load:

            kaggle.api.authenticate()

            kaggle.api.dataset_download_files('The_name_of_the_dataset', path='the_path_you_want_to_download_the_files_to', unzip=True)
        ## The direct link to the Kaggle data set
        #data_url = 'https://www.kaggle.com/crawford/gene-expression/downloads/actual.csv'

        ## The local path where the data set is saved.
        #local_filename = "actsual.csv"

        ## Kaggle Username and Password
        #kaggle_info = {'UserName': "myUsername", 'Password': "myPassword"}

        ## Attempts to download the CSV file. Gets rejected because we are not logged in.
        #r = requests.get(data_url)

        ## Login to Kaggle and retrieve the data.
        #r = requests.post(r.url, data = kaggle_info)

        ## Writes the data to a local file one chunk at a time.
        #f = open(local_filename, 'wb')
        #for chunk in r.iter_content(chunk_size = 512 * 1024): # Reads 512KB at a time into memory

        #    if chunk: # filter out keep-alive new chunks
        #        f.write(chunk)
        #f.close()