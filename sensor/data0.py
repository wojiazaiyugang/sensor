"""
处理data0的相关逻辑
"""
import os
import pickle
from typing import Tuple

import matplotlib.pyplot as plt

from settings import CYCLE_FILE_DIR, logger, DATA0_DIR, np


def load_data0_cycle() -> Tuple[np.ndarray, np.ndarray]:
    """
    载入用data0生成的步态周期数据
    :return:
    """
    data_file_full_name = os.path.join(CYCLE_FILE_DIR, "data")
    if not os.path.exists(data_file_full_name):
        logger.debug("data0 cycle数据不存在，开始生成")
        data, label = [], []
        from sensor.sensor import SensorManager
        from sensor.algorithm.data_pre_process import AccDataPreProcess, GyroDataPreProcess
        for i in range(10):
            sensor_manager = SensorManager(i)
            acc_data_pre_process = AccDataPreProcess(sensor_manager)
            gyro_data_pre_process = GyroDataPreProcess(sensor_manager)
            acc_cycles = []
            gyro_cycles = []
            while True:
                get_data_result = sensor_manager.update_display_raw_data()
                if not get_data_result:
                    break
                sensor_manager.acc_to_display, acc_cycle = acc_data_pre_process.get_gait_cycle(
                    sensor_manager.acc_to_display)
                sensor_manager.gyro_to_display, gyro_cycle = gyro_data_pre_process.get_gait_cycle(
                    sensor_manager.gyro_to_display)
                if acc_cycle is not None:
                    acc_cycles.append(acc_cycle)
                if gyro_cycle is not None:
                    gyro_cycles.append(gyro_cycle)
            for acc_cycle, gyro_cycle in zip(acc_cycles, gyro_cycles):
                data.append(np.concatenate((acc_cycle, gyro_cycle), axis=1))
                label.append(i)
            logger.debug("生成CNN数据：{0}".format(i))
        with open(data_file_full_name, "wb") as file:
            file.write(pickle.dumps((np.array(data), np.array(label))))
    with open(data_file_full_name, "rb") as file:
        data, label = pickle.loads(file.read())
    assert isinstance(data, np.ndarray) and len(data.shape) == 3 and data.shape[1] == 200 and data.shape[2] == 8
    return data.transpose((0, 2, 1)) , label  # 数据是200 * 8的，训练需要8 * 200


def load_data0_data(file_name: str) -> np.ndarray:
    """
    读data0的数据
    :param file_name:
    :return:
    """
    with open(file_name, "r", encoding="utf-8") as file:
        lines = file.readlines()
        lines = [[float(v) for v in line.split(" ")] for line in lines]
        return np.array(lines)
