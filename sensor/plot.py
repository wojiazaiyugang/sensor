from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# 支持中文
plt.rcParams["font.sans-serif"] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
from sensor.sensor import SensorManager


class PlotManager:
    """
    绘图管理
    """

    def __init__(self):
        # 传感器管理类，用于获得数据
        self.sensor_manager = SensorManager()
        # 数据图
        self.fig, (self.ax_acc, self.ax_gyro, self.ax_ang) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        self._, self._, self.figure_w, self.figure_h = self.fig.bbox.bounds
        self.figure_w = int(self.figure_w)
        self.figure_h = int(self.figure_h)
        # 加速度
        self.ax_acc.set_title("加速度")
        self.ax_acc.set_xlim(0, self.sensor_manager.ACC_POINT_COUNT)
        self.ax_acc.set_ylim(-12, 12)
        self.line_acc_x = self.ax_acc.plot([], [], "b-", label="x")[0]
        self.line_acc_y = self.ax_acc.plot([], [], "g-", label="y")[0]
        self.line_acc_z = self.ax_acc.plot([], [], "r-", label="z")[0]
        self.ax_acc.legend(loc="upper right")
        # 陀螺仪
        self.ax_gyro.set_title("陀螺仪")
        self.ax_gyro.set_xlim(0, self.sensor_manager.GYRO_POINT_COUNT)
        self.ax_gyro.set_ylim(-200, 200)
        self.line_gyro_x = self.ax_gyro.plot([], [], "b-", label="x")[0]
        self.line_gyro_y = self.ax_gyro.plot([], [], "g-", label="y")[0]
        self.line_gyro_z = self.ax_gyro.plot([], [], "r-", label="z")[0]
        self.ax_gyro.legend(loc="upper right")
        # 角度
        self.ax_ang.set_title("角度")
        self.ax_ang.set_xlim(0, self.sensor_manager.ANG_POINT_COUNT)
        self.ax_ang.set_ylim(-180, 180)
        self.line_ang_x = self.ax_ang.plot([], [], "b-", label="x")[0]
        self.line_ang_y = self.ax_ang.plot([], [], "g-", label="y")[0]
        self.line_ang_z = self.ax_ang.plot([], [], "r-", label="z")[0]
        self.ax_ang.legend(loc="upper right")

        self.sensor_manager.set_handler()
    def update(self):
        """
        更新曲线数据
        :return:
        """
        t = list(range(len(self.sensor_manager.acc_x[-self.sensor_manager.ACC_POINT_COUNT:])))
        self.line_acc_x.set_data(t, self.sensor_manager.acc_x[-self.sensor_manager.ACC_POINT_COUNT:])
        self.line_acc_y.set_data(t, self.sensor_manager.acc_y[-self.sensor_manager.ACC_POINT_COUNT:])
        self.line_acc_z.set_data(t, self.sensor_manager.acc_z[-self.sensor_manager.ACC_POINT_COUNT:])
        t = list(range(len(self.sensor_manager.gyro_x[-self.sensor_manager.GYRO_POINT_COUNT:])))
        self.line_gyro_x.set_data(t, self.sensor_manager.gyro_x[-self.sensor_manager.GYRO_POINT_COUNT:])
        self.line_gyro_y.set_data(t, self.sensor_manager.gyro_y[-self.sensor_manager.GYRO_POINT_COUNT:])
        self.line_gyro_z.set_data(t, self.sensor_manager.gyro_z[-self.sensor_manager.GYRO_POINT_COUNT:])
        t = list(range(len(self.sensor_manager.ang_x[-self.sensor_manager.ANG_POINT_COUNT:])))
        self.line_ang_x.set_data(t, self.sensor_manager.ang_x[-self.sensor_manager.ANG_POINT_COUNT:])
        self.line_ang_y.set_data(t, self.sensor_manager.ang_y[-self.sensor_manager.ANG_POINT_COUNT:])
        self.line_ang_z.set_data(t, self.sensor_manager.ang_z[-self.sensor_manager.ANG_POINT_COUNT:])
