from sensor.algorithm import AlgorithmManager
from settings import plt, np


class GaitFig:
    def __init__(self, algorithm_manager: AlgorithmManager):
        self.algorithm_manager = algorithm_manager
        self.gait_cycles = []
        self.fig, self.axs = plt.subplots(nrows=1, ncols=5, figsize=(5, 1))
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0)
        self._, self._, self.fig_gait_acc_w, self.fig_gait_acc_h = [int(i) for i in self.fig.bbox.bounds]
        for ax in self.axs:
            ax.set_title(str(ax))
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)

        self.gei_count_to_generate_geis = 30  # 使用多少张gei来生成geis

    def _get_gait_cycle(self):
        raise NotImplementedError

    def update_cycle_fig(self):
        gait_cycle = self._get_gait_cycle()
        if gait_cycle is not None:
            for index, ax in enumerate(self.axs[1:]):
                ax.cla()
                ax.plot(gait_cycle[:, index], color="black", linewidth=3)
            gei = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep="").reshape(
                self.fig.canvas.get_width_height()[::-1] + (3,))
            self.gait_cycles.append(gei)
        # self.fig.canvas.draw()

    def _get_template(self) -> np.ndarray:
        raise NotImplementedError

    def update_template_fig(self):
        """
        更新模板的曲线
        :return:
        """
        template = self._get_template()
        if template is not None:
            self.axs[0].cla()
            self.axs[0].plot(template)

    def get_gei(self):
        if not self.gait_cycles:
            return None
        return np.average(self.gait_cycles[-self.gei_count_to_generate_geis:], axis=0).astype("uint8")


class GaitAccFig(GaitFig):
    def _get_template(self) -> np.ndarray:
        return self.algorithm_manager.acc_data_pre_process.template

    def _get_gait_cycle(self):
        return self.algorithm_manager.last_acc_cycle


class GaitGyroFig(GaitFig):
    def _get_template(self) -> np.ndarray:
        return self.algorithm_manager.gyro_data_pre_process.template

    def _get_gait_cycle(self):
        return self.algorithm_manager.last_gyro_cycle


class GaitAngFig(GaitFig):
    def _get_template(self) -> np.ndarray:
        return self.algorithm_manager.ang_data_pre_process.template

    def _get_gait_cycle(self):
        return self.algorithm_manager.last_ang_cycle
