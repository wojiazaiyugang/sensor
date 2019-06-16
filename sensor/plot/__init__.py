from sensor.algorithm import AlgorithmManager
from sensor.plot.gait import GaitAccFig, GaitGyroFig
from sensor.plot.raw_data import RawDataFig, RawDataAccAxes, RawDataGyroAxes
from sensor.plot.stability import FigStability
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

        self.fig_raw_data = RawDataFig(sensor_manager)
        self.axes_raw_acc = RawDataAccAxes(self.fig_raw_data.ax_acc, sensor_manager)
        self.axes_raw_gyro = RawDataGyroAxes(self.fig_raw_data.ax_gyro, sensor_manager)

        self.fig_gait_acc = GaitAccFig(algorithm_manager)
        self.fig_gait_gyro = GaitGyroFig(algorithm_manager)

        self.fig_stability = FigStability(algorithm_manager)

    def update_display_raw_data_fig(self):
        """
        更新原始数据的图像
        :return:
        """
        self.axes_raw_acc.update()
        self.axes_raw_gyro.update()

    def update_gait_figure(self):
        """
        更新所有绘图区域
        :return:
        """
        self.fig_gait_acc.update()
        self.fig_gait_gyro.update()

