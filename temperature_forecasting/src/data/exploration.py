import matplotlib.pyplot as plt
import numpy as np


class Visualization:
    def __init__(self,df):
        self.df = df

    @staticmethod  # 声明为静态方法，可以直接通过类调用，不用创建实例 viz = Visualization() viz.plot()
    def plot(X: np.ndarray=None, y: np.ndarray=None, xlabel: str='', title: str=''):
        plt.plot(X=X)  # 24小时
        plt.plot(y=y)
        plt.xlabel(xlabel=xlabel)
        plt.title(title=title)
        plt.show()

    @staticmethod
    def plot_hist2d( X: np.ndarray=None, y: np.ndarray=None, xlabel: str=None, ylabel: str=None):
        plt.hist2d(X=X, y=y, bins=(50, 50), vmax=400)
        plt.colorbar()
        plt.xlabel(xlabel=xlabel)
        plt.ylabel(ylabel=ylabel)
        plt.show()



