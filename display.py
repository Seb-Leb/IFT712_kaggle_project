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

    def draw_scatter_plot(self, points, colors):
        plt.scatter(points[:,0], points[:,1], c=colors)
        plt.show()
