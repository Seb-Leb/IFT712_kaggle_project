import numpy as np
from matplotlib import pyplot as plt

class FigureDrawer:
    def __init__(self):
        pass

    def draw_hist_array(self, X, bins=1000):
        grid = int(np.sqrt(X.shape[0]))
        fig, axs = plt.subplots(grid, grid, facecolor='w')
        axs = axs.ravel()
        for n,x in enumerate(X):
            axs[n].hist(x, bins=bins)
        plt.show()

    def draw_line_plot(self, curves):
        plt.show()

    def draw_scatter_plot(self, points):
        plt.show()
