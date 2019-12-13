import numpy as np
from matplotlib import pyplot as plt

class FigureDrawer:
    def __init__(self, title, figsize):
        self.title = title
        self.figsize = figsize

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

    def draw_line_plot(self, lines, colors, labels, x_lab, y_lab):
        fig = plt.subplots(figsize=self.figsize)
        plt.title(self.title)
        for n,line in enumerate(lines):
            x, y = line
            plt.plot(x, y, c=colors[n], label=labels[n])
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.legend()
        plt.show()

    def draw_scatter_plot(self, points, titles):
        '''
        points is a tuble of 2d arrays.
        '''
        if len(points)>3:
            grid = int(np.sqrt(len(points)))
            fig, axs = plt.subplots(grid, grid, facecolor='w', figsize=self.figsize)

        fig, axs = plt.subplots(1, len(points), facecolor='w', figsize=self.figsize)
        axs = axs.ravel()
        for n,x in enumerate(points):
            axs[n].set_title(titles[n])
            axs[n].scatter(x[:,0], x[:,1], c=x[:,2], alpha=0.8, s=1.)
        plt.tight_layout()
        plt.show()

    def draw_box_plots(self, data, labels, y_lab):
        fig, ax = plt.subplots()
        ax.boxplot(data)
        ax.set_xticklabels(labels)
        plt.title(self.title)
        plt.ylabel(y_lab)
        plt.show()


