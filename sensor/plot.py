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


class PlotManager:
    """
    绘图管理
    """

    def __init__(self, sensor_manager: SensorManager, algorithm_manager:AlgorithmManager):
        """
        初始化
        :param sensor_manager: 需要使用传感器初始化，对绘图的一些组件进行初始化
        """
        # 传感器管理类，用于获得数据
        self.sensor_manager = sensor_manager
        # TODO
        self.algorithm_manager = algorithm_manager
        # 原始数据图 ===================================================================================================
        # 数据图
        self.fig_raw_data, (self.ax_acc, self.ax_gyro, self.ax_ang) = plt.subplots(nrows=3, ncols=1, figsize=(3, 9))
        self.fig_raw_data.subplots_adjust(left =0.1,bottom=0.005,right=1,top=1,hspace=0.1)
        self.ax_acc.get_xaxis().set_visible(False)
        self.ax_gyro.get_xaxis().set_visible(False)
        self.ax_ang.get_xaxis().set_visible(False)
        self._, self._, self.fig_raw_data_w, self.fig_raw_data_h = [int(i) for i in self.fig_raw_data.bbox.bounds]
        # 加速度
        # self.ax_acc.set_title("加速度")
        self.ax_acc.set_xlim(0, self.sensor_manager.ACC_POINT_COUNT)
        self.ax_acc.set_ylim(-15, 15)
        self.line_acc_x = self.ax_acc.plot([], [], "b-", label="acc_x")[0]
        self.line_acc_y = self.ax_acc.plot([], [], "g-", label="acc_y")[0]
        self.line_acc_z = self.ax_acc.plot([], [], "r-", label="acc_z")[0]
        self.ax_acc.legend(loc="upper right")
        # 陀螺仪
        # self.ax_gyro.set_title("陀螺仪")
        self.ax_gyro.set_xlim(0, self.sensor_manager.GYRO_POINT_COUNT)
        self.ax_gyro.set_ylim(-5, 5)
        self.line_gyro_x = self.ax_gyro.plot([], [], "b-", label="gyro_x")[0]
        self.line_gyro_y = self.ax_gyro.plot([], [], "g-", label="gyro_y")[0]
        self.line_gyro_z = self.ax_gyro.plot([], [], "r-", label="gyro_z")[0]
        self.ax_gyro.legend(loc="upper right")
        # 角度
        # self.ax_ang.set_title("角度")
        self.ax_ang.set_xlim(0, self.sensor_manager.ANG_POINT_COUNT)
        self.ax_ang.set_ylim(-180, 180)
        self.line_ang_x = self.ax_ang.plot([], [], "b-", label="ang_x")[0]
        self.line_ang_y = self.ax_ang.plot([], [], "g-", label="ang_y")[0]
        self.line_ang_z = self.ax_ang.plot([], [], "r-", label="ang_z")[0]
        self.ax_ang.legend(loc="upper right")
        # 加速度步态数据图 ===================================================================================================
        self.fig_acc_gait, (self.ax_gait_acc_mag, self.ax_gait_acc_x, self.ax_gait_acc_y, self.ax_gait_acc_z) = plt.subplots(nrows=1, ncols=4, figsize=(4, 1))
        self.fig_acc_gait.subplots_adjust(left =0, bottom=0, right=1, top=1, wspace=0)
        self._, self._, self.fig_gait_acc_w, self.fig_gait_acc_h = [int(i) for i in self.fig_acc_gait.bbox.bounds]
        self.ax_gait_acc_mag.set_title("mag")
        self.ax_gait_acc_mag.get_yaxis().set_visible(False)
        self.ax_gait_acc_mag.get_xaxis().set_visible(False)
        self.ax_gait_acc_x.set_title("x")
        self.ax_gait_acc_x.get_yaxis().set_visible(False)
        self.ax_gait_acc_x.get_xaxis().set_visible(False)
        self.ax_gait_acc_y.set_title("y")
        self.ax_gait_acc_y.get_yaxis().set_visible(False)
        self.ax_gait_acc_y.get_xaxis().set_visible(False)
        self.ax_gait_acc_z.set_title("z")
        self.ax_gait_acc_z.get_yaxis().set_visible(False)
        self.ax_gait_acc_z.get_xaxis().set_visible(False)
        # 陀螺仪步态数据图 ===================================================================================================
        self.fig_gyro_gait, (
        self.ax_gait_gyro_mag, self.ax_gait_gyro_x, self.ax_gait_gyro_y, self.ax_gait_gyro_z) = plt.subplots(nrows=1,
                                                                                                         ncols=4,
                                                                                                         figsize=(4, 1))
        self.fig_gyro_gait.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0)
        self._, self._, self.fig_gait_gyro_w, self.fig_gait_gyro_h = [int(i) for i in self.fig_gyro_gait.bbox.bounds]
        self.ax_gait_gyro_mag.set_title("mag")
        self.ax_gait_gyro_mag.get_yaxis().set_visible(False)
        self.ax_gait_gyro_mag.get_xaxis().set_visible(False)
        self.ax_gait_gyro_x.set_title("x")
        self.ax_gait_gyro_x.get_yaxis().set_visible(False)
        self.ax_gait_gyro_x.get_xaxis().set_visible(False)
        self.ax_gait_gyro_y.set_title("y")
        self.ax_gait_gyro_y.get_yaxis().set_visible(False)
        self.ax_gait_gyro_y.get_xaxis().set_visible(False)
        self.ax_gait_gyro_z.set_title("z")
        self.ax_gait_gyro_z.get_yaxis().set_visible(False)
        self.ax_gait_gyro_z.get_xaxis().set_visible(False)

        self.count_threshold_clear = 400 # 阈值，超过这个阈值还没有生成步态就认为数据有问题，直接清除数据
        # 预处理之后的数据，用于显示绘图
        self.acc_data = None
        self.gyro_data = None
        self.gei_count_to_generate_geis = 30 # 使用多少张gei来生成geis
        #geis图像
        self.acc_gei = None
        self.gyro_gei = None

    def update_raw_data(self):
        """
        更新曲线数据
        :return:
        """
        acc_data = numpy.array(self.sensor_manager.acc[-self.sensor_manager.ACC_POINT_COUNT:])
        if acc_data.any():
            validate_raw_data_with_timestamp(acc_data)
            t = list(range(len(acc_data)))
            self.line_acc_x.set_data(t, acc_data[:, 1])
            self.line_acc_y.set_data(t, acc_data[:, 2])
            self.line_acc_z.set_data(t, acc_data[:, 3])
        gyro_data = numpy.array(self.sensor_manager.gyro[-self.sensor_manager.GYRO_POINT_COUNT:])
        if gyro_data.any():
            validate_raw_data_with_timestamp(gyro_data)
            t = list(range(len(gyro_data)))
            self.line_gyro_x.set_data(t, gyro_data[:, 1])
            self.line_gyro_y.set_data(t, gyro_data[:, 2])
            self.line_gyro_z.set_data(t, gyro_data[:, 3])
        ang_data = numpy.array(self.sensor_manager.ang[-self.sensor_manager.ANG_POINT_COUNT:])
        if ang_data.any():
            validate_raw_data_with_timestamp(ang_data)
            t = list(range(len(ang_data)))
            self.line_ang_x.set_data(t, ang_data[:, 1])
            self.line_ang_y.set_data(t, ang_data[:, 2])
            self.line_ang_z.set_data(t, ang_data[:, 3])

    def update_figures(self):
        """
        更新所有绘图区域
        :return:
        """
        # 原始数据
        if SENSOR_DATA is not None:  # 不是实时数据的话，需要先去模拟一波数据
            self.sensor_manager.mock_real_time_data_from_data0()
        self.update_raw_data()

        color = "black"
        linewidth = 3
        # 加速度步态数据
        self.acc_data = self.algorithm_manager.data_pre_process.pre_process(numpy.array(self.sensor_manager.acc), "acc")
        if self.acc_data is not None:
            self.ax_gait_acc_mag.cla()
            self.ax_gait_acc_mag.plot(self.acc_data[:, 0], color=color, linewidth=linewidth)
            self.ax_gait_acc_x.cla()
            self.ax_gait_acc_x.plot(self.acc_data[:, 1], color=color, linewidth=linewidth)
            self.ax_gait_acc_y.cla()
            self.ax_gait_acc_y.plot(self.acc_data[:, 2], color=color, linewidth=linewidth)
            self.ax_gait_acc_z.cla()
            self.ax_gait_acc_z.plot(self.acc_data[:, 3], color=color, linewidth=linewidth)
            self.sensor_manager.acc.clear()
            self.fig_acc_gait.canvas.draw()
            gei = numpy.fromstring(self.fig_acc_gait.canvas.tostring_rgb(), dtype=numpy.uint8, sep="").reshape(
                self.fig_acc_gait.canvas.get_width_height()[::-1] + (3,))
            self.algorithm_manager.data_pre_process.acc_geis.append(gei)
            self.acc_gei = numpy.average(self.algorithm_manager.data_pre_process.acc_geis[-self.gei_count_to_generate_geis:], axis=0).astype("uint8")
        # 清除数据
        if self.acc_data is not None or len(self.sensor_manager.acc) > self.count_threshold_clear:
            self.sensor_manager.acc.clear()
        self.gyro_data = self.algorithm_manager.data_pre_process.pre_process(numpy.array(self.sensor_manager.gyro), "gyro")
        # 步态数据
        if self.gyro_data is not None:

            self.ax_gait_gyro_mag.cla()
            self.ax_gait_gyro_mag.plot(self.gyro_data[:, 0], color=color, linewidth=linewidth)
            self.ax_gait_gyro_x.cla()
            self.ax_gait_gyro_x.plot(self.gyro_data[:, 1], color=color, linewidth=linewidth)
            self.ax_gait_gyro_y.cla()
            self.ax_gait_gyro_y.plot(self.gyro_data[:, 2], color=color, linewidth=linewidth)
            self.ax_gait_gyro_z.cla()
            self.ax_gait_gyro_z.plot(self.gyro_data[:, 3], color=color, linewidth=linewidth)
            self.sensor_manager.gyro.clear()
            self.fig_gyro_gait.canvas.draw()
            gei = numpy.fromstring(self.fig_gyro_gait.canvas.tostring_rgb(), dtype=numpy.uint8, sep="").reshape(
                self.fig_gyro_gait.canvas.get_width_height()[::-1] + (3,))
            self.algorithm_manager.data_pre_process.gyro_geis.append(gei)
            self.gyro_gei = numpy.average(
                self.algorithm_manager.data_pre_process.gyro_geis[-self.gei_count_to_generate_geis:], axis=0).astype(
                "uint8")
        # 清除数据
        if self.gyro_data is not None or len(self.sensor_manager.gyro) > self.count_threshold_clear:
            self.sensor_manager.gyro.clear()
