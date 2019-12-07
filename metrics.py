



#####    Kaggle project
# Jeremie Beliveau-Lefebvre (04494470)
# Sebastien Leblanc         (18206273)
###



import numpy as np
import matplotlib.pyplot as plt



class metrics :
    def __init__(self, nb_train, nb_test, lineairement_sep):
        self.nb_train = nb_train
        self.nb_test = nb_test
        self.lineairement_sep = lineairement_sep

    def affichage(self, x_tab, t_tab):
        # Affichage
        ix = np.arange(x_tab[:, 0].min(), x_tab[:, 0].max(), 0.1)
        iy = np.arange(x_tab[:, 1].min(), x_tab[:, 1].max(), 0.1)
        iX, iY = np.meshgrid(ix, iy)
        x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
        contour_out = np.array([self.prediction(x) for x in x_vis])
        contour_out = contour_out.reshape(iX.shape)

        plt.contourf(iX, iY, contour_out > 0.5)
        plt.scatter(x_tab[:, 0], x_tab[:, 1], s=(t_tab + 0.5) * 100, c=t_tab, edgecolors='y')

        plt.title('Erreurs')     
        plt.show()
