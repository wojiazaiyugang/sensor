"""
画图支持
"""
import numpy
import matplotlib.pyplot as plt

# 支持中文
plt.rcParams["font.sans-serif"] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
from sensor.sensor import SensorManager
from sensor.algorithm import AlgorithmManager
from util import validate_raw_data_with_timestamp

from settings import SENSOR_DATA, logger


class RawDataFig:
    def __init__(self, sensor_manager: SensorManager):
        self.sensor_manager = sensor_manager
        # 数据图
        self.fig, (self.ax_acc, self.ax_gyro, self.ax_ang) = plt.subplots(nrows=3, ncols=1, figsize=(3, 9))
        self.fig.subplots_adjust(left=0.1, bottom=0.005, right=1, top=1, hspace=0.1)
        self._, self._, self.width, self.height = [int(i) for i in self.fig.bbox.bounds]


class RawDataAxes:
    def __init__(self, ax, sensor_manager: SensorManager):
        self.data_type = self.data_type
        self.sensor_manager = sensor_manager
        ax.get_xaxis().set_visible(False)
        ax.set_xlim(0, 400)
        self.line_x = ax.plot([], [], "b-", label="{0}_x".format(self.data_type))[0]
        self.line_y = ax.plot([], [], "g-", label="{0}_y".format(self.data_type))[0]
        self.line_z = ax.plot([], [], "r-", label="{0}_z".format(self.data_type))[0]
        ax.legend(loc="upper right")

    def get_raw_data(self):
        pass

    def update_raw_data(self):
        """
        更新曲线数据
        :return:
        """
        acc_data = numpy.array(self.get_raw_data()[-400:])
        if acc_data.any():
            validate_raw_data_with_timestamp(acc_data)
            t = list(range(len(acc_data)))
            self.line_x.set_data(t, acc_data[:, 1])
            self.line_y.set_data(t, acc_data[:, 2])
            self.line_z.set_data(t, acc_data[:, 3])


class RawDataAccAxes(RawDataAxes):
    def __init__(self, ax, sensor_manager):
        self.data_type = "acc"
        ax.set_ylim(-15, 15)
        super().__init__(ax, sensor_manager)

    def get_raw_data(self):
        return self.sensor_manager.acc


class RawDataGyroAxes(RawDataAxes):
    def __init__(self, ax, sensor_manager):
        self.data_type = "gyro"
        ax.set_ylim(-15, 15)
        super().__init__(ax, sensor_manager)

    def get_raw_data(self):
        return self.sensor_manager.gyro


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
            for index,ax in enumerate(self.axs):
                ax.cla()
                ax.plot(gait_cycle[:, index], color="black", linewidth=3)
            self.fig.canvas.draw()
            gei = numpy.fromstring(self.fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep="").reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
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


class PlotManager:
    """
    绘图管理
    """

    def __init__(self, sensor_manager: SensorManager, algorithm_manager: AlgorithmManager):
        """
        初始化
        :param sensor_manager: 需要使用传感器初始化，对绘图的一些组件进行初始化
        """
        # 传感器管理类，用于获得数据
        self.sensor_manager = sensor_manager
        self.algorithm_manager = algorithm_manager

        self.DEBUG = None
        self.raw_data_fig = RawDataFig(sensor_manager)
        self.raw_data_acc_axes = RawDataAccAxes(self.raw_data_fig.ax_acc, sensor_manager)
        self.raw_data_gyro_axes = RawDataGyroAxes(self.raw_data_fig.ax_gyro, sensor_manager)

        self.gait_acc_fig = GaitAccFig(algorithm_manager)
        self.gait_gyro_fig = GaitGyroFig(algorithm_manager)

    def update_figures(self):
        """
        更新所有绘图区域
        :return:
        """
        # 原始数据
        if SENSOR_DATA is not None:  # 不是实时数据的话，需要先去模拟一波数据
            self.sensor_manager.mock_real_time_data_from_data0()
        self.raw_data_acc_axes.update_raw_data()
        self.raw_data_gyro_axes.update_raw_data()

        self.gait_acc_fig.update_cycle_fig()
        self.gait_gyro_fig.update_cycle_fig()
