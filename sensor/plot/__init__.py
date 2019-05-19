from sensor.sensor import SensorManager
from sensor.algorithm import AlgorithmManager
from sensor.plot.raw_data import RawDataFig, RawDataAccAxes, RawDataGyroAxes
from sensor.plot.gait import GaitAccFig, GaitGyroFig
from settings import SENSOR_DATA

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