from sensor.algorithm import AlgorithmManager
from settings import plt, np


class GaitFig:
    def __init__(self, algorithm_manager: AlgorithmManager):
        self.algorithm_manager = algorithm_manager
        self.gait_cycles = []
        self.gei = None
        self.fig, self.axs = plt.subplots(nrows=1, ncols=5, figsize=(5, 1))
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0)
        self._, self._, self.fig_width, self.fig_height = [int(i) for i in self.fig.bbox.bounds]
        for ax in self.axs:
            ax.set_title(str(ax))
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
        self.count_to_generate_gei = 10  # 使用多少张gei来生成geis

    def _get_gait_cycle(self):
        raise NotImplementedError

    def update(self):
        """
        更新图
        :return:
        """
        template = self._get_template()
        if template is not None:
            self.axs[0].cla()
            self.axs[0].plot(template)
        gait_cycle = self._get_gait_cycle()
        if gait_cycle is not None:
            for index, ax in enumerate(self.axs[1:]):
                ax.cla()
                ax.plot(gait_cycle[:, index], color="black", linewidth=3)
            self.fig.canvas.draw()
            gei = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep="").reshape(
                self.fig.canvas.get_width_height()[::-1] + (3,))
            self.gait_cycles.append(gei)
            self.gait_cycles = self.gait_cycles[-self.count_to_generate_gei:]

            self.gei = np.average(self.gait_cycles, axis=0).astype("uint8")

    def _get_template(self) -> np.ndarray:
        """
        获取模板
        :return:
        """
        raise NotImplementedError


class GaitAccFig(GaitFig):
    def _get_template(self) -> np.ndarray:
        return self.algorithm_manager.acc_data_pre_process.template

    def _get_gait_cycle(self):
        return self.algorithm_manager.acc_data_pre_process.last_cycle


class GaitGyroFig(GaitFig):
    def _get_template(self) -> np.ndarray:
        return self.algorithm_manager.gyro_data_pre_process.template

    def _get_gait_cycle(self):
        return self.algorithm_manager.gyro_data_pre_process.last_cycle

