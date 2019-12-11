import numpy as np
from matplotlib import pyplot as plt

class FigureDrawer:
    def __init__(self, title):
        self.title = title

    def draw_hist_array(self, X, labels, bins=1000):
        grid = int(np.sqrt(X.shape[0]))
        fig, axs = plt.subplots(grid, grid, facecolor='w')
        axs = axs.ravel()
        for n,x in enumerate(X):
            axs[n].set_title(labels[n])
            axs[n].hist(x, bins=bins)
            axs[n].set_yscale('log')
        plt.tight_layout()
        plt.show()

    def draw_line_plot(self, curves):
        plt.show()

    def draw_scatter_plot(self, points, labels, figsize):
        '''
        points is a tuble of 2d arrays.
        '''
        if len(points)>3:
            grid = int(np.sqrt(len(points)))
            fig, axs = plt.subplots(grid, grid, facecolor='w')

        fig, axs = plt.subplots(1, len(points), facecolor='w', figsize=figsize)
        axs = axs.ravel()
        for n,x in enumerate(points):
            axs[n].set_title(labels[n])
            axs[n].scatter(x[:,0], x[:,1], c=x[:,2], alpha=0.8, s=1.)
        plt.tight_layout()
        plt.show()
