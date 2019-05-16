"""
传感器数据相关支持
"""
import os
import math

from numpy import short
import pywinusb.hid as hid

from util import get_current_timestamp, get_data0_data
from settings import SENSOR_DATA, DATA_DIR, logger


class SensorManager:
    """
    传感器管理
    """

    def __init__(self):
        # 传感器，用于实时数据
        self.sensor = None
        # 模拟数据
        self.last_data_index = 0 # 上一次载入的数据的index
        self.last_data_timestamp = None  # 传感器开始时间，用于模拟数据的时候读取数据
        self.acc_data_lines = None  # 模拟数据
        self.gyro_data_lines = None  # 模拟数据
        # 最多保存的数据点的个数
        self.ACC_POINT_COUNT = 400
        self.GYRO_POINT_COUNT = 400
        self.ANG_POINT_COUNT = 400
        # 加速度数据
        self.acc = []
        # 陀螺仪数据
        self.gyro = []
        # 角度
        self.ang = []
        self.set_handler()

    def _on_get_data(self, data: list):
        """
        传感器数据回调函数
        :param data: 数据
        :return:
        """
        for i in range(len(data) - 10):
            if data[i] == 0x55 and data[i + 1] == 0x51:
                axl, axh, ayl, ayh, azl, azh, *_ = data[i + 2:i + 11]
                self.acc.append([get_current_timestamp(), (short((axh << 8) | axl)) / 32768 * 16 * 9.8,
                                 (short((ayh << 8) | ayl)) / 32768 * 16 * 9.8,
                                 (short((azh << 8) | azl)) / 32768 * 16 * 9.8])
                self._validate_raw_data(self.acc)

            if data[i] == 0x55 and data[i + 1] == 0x52:
                wxl, wxh, wyl, wyh, wzl, wzh, *_ = data[i + 2:i + 11]
                self.gyro.append([get_current_timestamp(), (short(wxh << 8) | wxl) / 32768 * 2000 * (math.pi / 180),
                                  (short(wyh << 8) | wyl) / 32768 * 2000 * (math.pi / 180), (short(wzh << 8) | wzl) / 32768 * 2000* (math.pi / 180)])
                # TODO: 这里每两个数就有一个是0，目前不知道为什么，影响效果先去除
                # if len(self.gyro) > 2 and self.gyro[-1][1] and not self.gyro[-2][1] and self.gyro[-3][1]:
                #     del self.gyro[-1]
                self._validate_raw_data(self.gyro)
            # if data[i] == 0x55 and data[i + 1] == 0x53:
            #     rol, roh, pil, pih, yal, yah, *_ = data[i + 2:i + 11]
            #     self.ang.append([get_current_timestamp(), (short(roh << 8 | rol) / 32768 * 180),
            #                      (short(pih << 8 | pil) / 32768 * 180), (short(yah << 8 | yal) / 32768 * 180)])

    @staticmethod
    def _get_sensor():
        """
        获取传感器
        :return: sensor
        """
        # 获取传感器hid
        hid_devices = hid.core.find_all_hid_devices()
        sensor = None
        for hid_device in hid_devices:
            if "Wit-Motion" in hid_device.product_name:
                sensor = hid_device
        if not sensor:
            raise Exception("没有设备")
        return sensor

    def set_handler(self):
        """
        注册回调函数，用于生成传感器数据，根据settings中的SENSOR_DATA来决定是使用实时数据还是使用data0数据
        :return:
        """
        logger.info("是否使用实时数据：{0}".format(bool(SENSOR_DATA is None)))
        if SENSOR_DATA is None:  # 这里必须是is None,因为SENSOR_DATA有可能是0
            self.sensor = self._get_sensor()
            # 打开设备
            self.sensor.open()
            # 注册回调函数
            self.sensor.set_raw_data_handler(self._on_get_data)
        else:
            logger.info("载入data0加速度数据")
            self.acc_data_lines = get_data0_data(os.path.join(DATA_DIR, "data0", "accData{0}.txt".format(SENSOR_DATA)))
            logger.info("载入data0陀螺仪数据")
            self.gyro_data_lines = get_data0_data(os.path.join(DATA_DIR, "data0", "gyrData{0}.txt".format(SENSOR_DATA)))
            self.last_data_timestamp = get_current_timestamp()

    def mock_real_time_data_from_data0(self):
        """
        使用data0中的数据模拟真实数据，做法是通过时间戳来决定读取数据的多少
        :return:
        """
        current_timestamp = get_current_timestamp()
        mock_data_count = (current_timestamp - self.last_data_timestamp) // 5 # // 20 # data0中的数据是采样频率50HZ
        current_data_index = self.last_data_index + mock_data_count
        self.acc.extend(self.acc_data_lines[self.last_data_index: current_data_index])
        self.gyro.extend(self.gyro_data_lines[self.last_data_index: current_data_index])
        self.last_data_timestamp = current_timestamp
        self.last_data_index = current_data_index

    def _validate_raw_data(self, data):
        threshold = 5
        if len(data) > 1 and abs(data[-1][1] - data[-2][1]) > threshold:
            logger.debug("EE")
            del data[-1]
        if len(data) > 1 and abs(data[-1][2] - data[-2][2]) > threshold:
            logger.debug("EE")
            del data[-1]
        if len(data) > 1 and abs(data[-1][3] - data[-2][3]) > threshold:
            logger.debug("EE")
            del data[-1]
