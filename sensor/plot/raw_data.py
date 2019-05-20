import numpy

from sensor.sensor import SensorManager
from util import validate_raw_data_with_timestamp
from settings import plt


class RawDataFig:
    def __init__(self, sensor_manager: SensorManager):
        self.sensor_manager = sensor_manager
        # 数据图
        self.fig, (self.ax_acc, self.ax_gyro, self.ax_ang) = plt.subplots(nrows=3, ncols=1, figsize=(3, 9))
        self.fig.subplots_adjust(left=0.1, bottom=0.005, right=1, top=1, hspace=0.1)
        self._, self._, self.width, self.height = [int(i) for i in self.fig.bbox.bounds]


class _RawDataAxes:
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


class RawDataAccAxes(_RawDataAxes):
    def __init__(self, ax, sensor_manager):
        self.data_type = "acc"
        ax.set_ylim(-15, 15)
        super().__init__(ax, sensor_manager)

    def get_raw_data(self):
        return self.sensor_manager.acc


class RawDataGyroAxes(_RawDataAxes):
    def __init__(self, ax, sensor_manager):
        self.data_type = "gyro"
        ax.set_ylim(-15, 15)
        super().__init__(ax, sensor_manager)

    def get_raw_data(self):
        return self.sensor_manager.gyro


class RawDataAngAxes(_RawDataAxes):
    def __init__(self, ax, sensor_manager):
        self.data_type = "ang"
        ax.set_ylim(-180, 180)
        super().__init__(ax, sensor_manager)

    def get_raw_data(self):
        return self.sensor_manager.ang
