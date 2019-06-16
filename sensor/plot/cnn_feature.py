from sensor.algorithm import AlgorithmManager
from settings import plt


class FigCnnFeature:
    def __init__(self, algorithm_manager: AlgorithmManager):
        self.algorithm_manager = algorithm_manager
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        self._, self._, self.fig_width, self.fig_height = [int(i) for i in self.fig.bbox.bounds]

    def update(self):
        pass
