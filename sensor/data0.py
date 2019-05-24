"""
处理data0的相关逻辑
"""
import os
import pickle
from typing import Tuple

import numpy
import matplotlib.pyplot as plt

from settings import CYCLE_FILE_DIR, logger, DATA0_DIR


def load_data0_cycle() -> Tuple[numpy.ndarray, numpy.ndarray]:
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
        acc_data_pre_process = AccDataPreProcess()
        gyro_data_pre_process = GyroDataPreProcess()
        for i in range(10):
            sensor_manager = SensorManager(i)
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
                data.append(numpy.concatenate((acc_cycle, gyro_cycle), axis=1))
                label.append(i)
            logger.debug("生成CNN数据：{0}".format(i))
        with open(data_file_full_name, "wb") as file:
            file.write(pickle.dumps((numpy.array(data), numpy.array(label))))
    with open(data_file_full_name, "rb") as file:
        data, label = pickle.loads(file.read())
    assert isinstance(data, numpy.ndarray) and len(data.shape) == 3 and data.shape[1] == 200 and data.shape[2] == 8
    return data, label


def load_data0_data(file_name: str) -> numpy.ndarray:
    """
    读data0的数据
    :param file_name:
    :return:
    """
    with open(file_name, "r", encoding="utf-8") as file:
        lines = file.readlines()
        lines = [[float(v) for v in line.split(" ")] for line in lines]
        return numpy.array(lines)


if __name__ == "__main__":
    data = load_data0_data(os.path.join(DATA0_DIR, "accData2.txt"))
    mag = [d[1] * d[1] + d[2] * d[2] + d[3] * d[3] for d in data]
    plt.plot(mag)
    plt.show()