"""
传感器数据相关支持
"""
import os
from typing import Union

import math
import pywinusb.hid as hid
from numpy import short

from sensor.data0 import load_data0_data
from settings import DATA_DIR, logger
from util import get_current_timestamp


class SensorManager:
    """
    传感器管理
    """

    def __init__(self, sensor_data: Union[int, None] = None):
        """
        初始化
        :param sensor_data: 数据类型，0 - 9 表示使用data0中的数据进行模拟，None表示使用实时数据
        """
        # 传感器，用于实时数据
        self.sensor = None
        # 模拟数据
        self.sensor_data = sensor_data
        # 最多保存的数据点的个数
        self.ACC_POINT_COUNT = 400
        self.GYRO_POINT_COUNT = 400
        self.ANG_POINT_COUNT = 400
        # 原始数据，显示使用
        self.acc_to_display = []
        self.gyro_to_display = []
        self.ang_to_display = []
        # 用于检测步态的数据
        self.acc_to_detect_cycle = []
        self.gyro_to_detect_cycle = []
        self.ang_to_detect_cycle = []

        logger.info("是否使用实时数据：{0}".format(bool(sensor_data is None)))
        if sensor_data is not None:
            assert sensor_data is None or 0 <= int(sensor_data) <= 9, "数据错误"
            self.last_data_index = 0  # 上一次载入的数据的index
            self.last_data_timestamp = None  # 传感器开始时间，用于模拟数据的时候读取数据
            self.acc_data_lines = None  # 模拟数据
            self.gyro_data_lines = None  # 模拟数据
            logger.info("载入data0加速度数据")
            self.acc_data_lines = load_data0_data(os.path.join(DATA_DIR, "data0", "accData{0}.txt".format(sensor_data)))
            logger.info("载入data0陀螺仪数据")
            self.gyro_data_lines = load_data0_data(
                os.path.join(DATA_DIR, "data0", "gyrData{0}.txt".format(sensor_data)))
            self.last_data_timestamp = get_current_timestamp()
        else:
            self.set_handler()

    def _on_get_data(self, data: list):
        """
        传感器数据回调函数
        :param data: 数据
        :return:
        """
        for i in range(len(data) - 11):
            if not (data[i] == 0x55 and data[i + 1] & 0x50 == 0x50):
                continue
            if data[i] == 0x55 and data[i + 1] == 0x51 and sum(data[i:i + 10]) & 255 == data[i + 10]:
                axl, axh, ayl, ayh, azl, azh, *_ = data[i + 2:i + 11]
                sensor_data = [
                    get_current_timestamp(),
                    (short((axh << 8) | axl)) / 32768 * 16 * 9.8,
                    (short((ayh << 8) | ayl)) / 32768 * 16 * 9.8,
                    (short((azh << 8) | azl)) / 32768 * 16 * 9.8
                ]
                self.acc_to_display.append(sensor_data)
                self.acc_to_detect_cycle.append(sensor_data)
            if data[i] == 0x55 and data[i + 1] == 0x52 and sum(data[i:i + 10]) & 255 == data[i + 10]:
                wxl, wxh, wyl, wyh, wzl, wzh, *_ = data[i + 2:i + 11]
                sensor_data = [
                    get_current_timestamp(),
                    (short(wxh << 8) | wxl) / 32768 * 2000 * (math.pi / 180),
                    (short(wyh << 8) | wyl) / 32768 * 2000 * (math.pi / 180),
                    (short(wzh << 8) | wzl) / 32768 * 2000 * (math.pi / 180)
                ]
                self.gyro_to_display.append(sensor_data)
                self.gyro_to_detect_cycle.append(sensor_data)
            if data[i] == 0x55 and data[i + 1] == 0x53 and sum(data[i:i + 10]) & 255 == data[i + 10]:
                rol, roh, pil, pih, yal, yah, *_ = data[i + 2:i + 11]
                sensor_data = [
                    get_current_timestamp(),
                    (short(roh << 8 | rol) / 32768 * 180),
                    (short(pih << 8 | pil) / 32768 * 180),
                    (short(yah << 8 | yal) / 32768 * 180)
                ]
                self.ang_to_display.append(sensor_data)
                self.acc_to_detect_cycle.append(sensor_data)
        self.fix_data_count()

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
        self.sensor = self._get_sensor()
        # 打开设备
        self.sensor.open()
        # 注册回调函数
        self.sensor.set_raw_data_handler(self._on_get_data)

    def _mock_real_time_data_from_data0(self) -> bool:
        """
        使用data0中的数据模拟真实数据
        :return:
        """
        mock_data_count = 20
        current_data_index = self.last_data_index + mock_data_count
        acc_mock_data = self.acc_data_lines[self.last_data_index: current_data_index]
        self.acc_to_display.extend(acc_mock_data)
        self.acc_to_detect_cycle.extend(acc_mock_data)
        gyro_mock_data = self.gyro_data_lines[self.last_data_index: current_data_index]
        self.gyro_to_display.extend(gyro_mock_data)
        self.gyro_to_detect_cycle.extend(gyro_mock_data)
        self.last_data_index = current_data_index
        self.fix_data_count()
        if current_data_index >= min(len(self.acc_data_lines), len(self.gyro_data_lines)):
            return False
        return True

    def update_display_raw_data(self) -> bool:
        """
        更新传感器原始数据。返回是否更新成功。如果是模拟数据并且使用完了就会返回失败,如果是实时数据就会一直成功
        :return:
        """
        if self.sensor_data is not None:
            return self._mock_real_time_data_from_data0()
        else:
            return True

    def clear_data_to_detect_cycle(self):
        """
        未检测到步行的时候，用于检测步态的数据进行清零
        :return:
        """
        self.acc_to_detect_cycle.clear()
        self.gyro_to_detect_cycle.clear()
        self.ang_to_detect_cycle.clear()

    def fix_data_count(self):
        """
        限制原始数据最大值，否则一直append就崩了
        :return:
        """
        self.acc_to_display = self.acc_to_display[-self.ACC_POINT_COUNT:]
        self.acc_to_detect_cycle = self.acc_to_detect_cycle[-self.ACC_POINT_COUNT:]
        self.gyro_to_display = self.gyro_to_display[-self.GYRO_POINT_COUNT:]
        self.gyro_to_detect_cycle = self.gyro_to_detect_cycle[-self.GYRO_POINT_COUNT:]
        self.ang_to_display = self.ang_to_display[-self.ANG_POINT_COUNT:]
        self.ang_to_detect_cycle = self.ang_to_detect_cycle[-self.ANG_POINT_COUNT:]
