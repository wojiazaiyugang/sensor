from sensor.algorithm import AlgorithmManager
from settings import plt


class FigStability:
    def __init__(self, algorithm_manager: AlgorithmManager):
        self.algorithm_manager = algorithm_manager
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        self._, self._, self.fig_w, self.fig_h = [int(i) for i in self.fig.bbox.bounds]
        self.ax.get_yaxis().set_visible(False)
        self.ax.get_xaxis().set_visible(False)
