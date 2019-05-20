import numpy

from sensor.algorithm import AlgorithmManager
from settings import plt


class GaitFig:
    def __init__(self, algorithm_manager: AlgorithmManager):
        self.algorithm_manager = algorithm_manager
        self.gait_cycles = []
        self.fig, self.axs = plt.subplots(nrows=1, ncols=4, figsize=(4, 1))
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0)
        self._, self._, self.fig_gait_acc_w, self.fig_gait_acc_h = [int(i) for i in self.fig.bbox.bounds]
        for ax in self.axs:
            ax.set_title(str(ax))
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)

        self.gei_count_to_generate_geis = 30  # 使用多少张gei来生成geis

    def get_gait_cycle(self):
        pass

    def update_cycle_fig(self):
        gait_cycle = self.get_gait_cycle()
        if gait_cycle is not None:
            for index, ax in enumerate(self.axs):
                ax.cla()
                ax.plot(gait_cycle[:, index], color="black", linewidth=3)
            self.fig.canvas.draw()
            gei = numpy.fromstring(self.fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep="").reshape(
                self.fig.canvas.get_width_height()[::-1] + (3,))
            self.gait_cycles.append(gei)

    def get_gei(self):
        if not self.gait_cycles:
            return None
        return numpy.average(self.gait_cycles[-self.gei_count_to_generate_geis:], axis=0).astype("uint8")


class GaitAccFig(GaitFig):
    def get_gait_cycle(self):
        return self.algorithm_manager.get_acc_gait_cycle()


class GaitGyroFig(GaitFig):
    def get_gait_cycle(self):
        return self.algorithm_manager.get_gyro_gait_cycle()


class GaitAngFig(GaitFig):
    def get_gait_cycle(self):
        return self.algorithm_manager.get_ang_gait_cycle()
