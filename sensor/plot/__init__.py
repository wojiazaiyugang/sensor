from sensor.algorithm import AlgorithmManager
from sensor.plot.gait import GaitAccFig, GaitGyroFig, GaitAngFig
from sensor.plot.raw_data import RawDataFig, RawDataAccAxes, RawDataGyroAxes, RawDataAngAxes
from sensor.sensor import SensorManager


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

        self.raw_data_fig = RawDataFig(sensor_manager)
        self.raw_data_acc_axes = RawDataAccAxes(self.raw_data_fig.ax_acc, sensor_manager)
        self.raw_data_gyro_axes = RawDataGyroAxes(self.raw_data_fig.ax_gyro, sensor_manager)
        self.raw_data_ang_axes = RawDataAngAxes(self.raw_data_fig.ax_ang, sensor_manager)

        self.gait_acc_fig = GaitAccFig(algorithm_manager)
        self.gait_gyro_fig = GaitGyroFig(algorithm_manager)
        self.gait_ang_fig = GaitAngFig(algorithm_manager)

    def update_display_raw_data_fig(self):
        """
        更新原始数据的图像
        :return:
        """
        self.raw_data_acc_axes.update_raw_data()
        self.raw_data_gyro_axes.update_raw_data()
        self.raw_data_ang_axes.update_raw_data()

    def update_gait_figure(self):
        """
        更新所有绘图区域
        :return:
        """
        self.gait_acc_fig.update_cycle_fig()
        self.gait_gyro_fig.update_cycle_fig()
        self.gait_ang_fig.update_cycle_fig()

    def update_template_fig(self):
        self.gait_acc_fig.update_template_fig()