



#####    Kaggle project
# Jeremie Beliveau-Lefebvre (04494470)
# Sebastien Leblanc         (18206273)
###


import numpy as np


def main():

    if len(sys.argv) < 6:
        usage = "\n Usage: python non_lineaire_classification.py type_noyau nb_train nb_test lin validation\
        \n\n\t type_noyau: rbf, lineaire, polynomial, sigmoidal\
        \n\t nb_train, nb_test: nb de donnees d'entrainement et de test\
        \n\t lin : 0: donnees non lineairement separables, 1: donnees lineairement separable\
        \n\t validation: 0: pas de validation croisee,  1: validation croisee\n"
        print(usage)
        return

    #type_noyau = sys.argv[1]
    #nb_train = int(sys.argv[2])
    #nb_test = int(sys.argv[3])
    #lin_sep = int(sys.argv[4])
    #vc = bool(int(sys.argv[5]))



if __name__ == "__main__":
    main()
