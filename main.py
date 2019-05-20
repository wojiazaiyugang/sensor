import os
import time
from sensor.algorithm import AlgorithmManager
from sensor.gui import GuiManager
from sensor.sensor import SensorManager
from settings import DATA_DIR
from util import get_data0_data

DEBUG = False


def convert_data0_to_gait_cycles_file(people_index: int):
    """
    把data0的数据进行周期分割，然后存进文件里供后续CNN等使用
    :param people_index:
    :return:
    """


if __name__ == "__main__":
    if not DEBUG:
        GuiManager().run()
    sensor_manager = SensorManager()
    algorithm_manager = AlgorithmManager(sensor_manager)
    algorithm_manager.gyro_data_pre_process.DEBUG = True
    while True:
        sensor_manager.mock_real_time_data_from_data0()
        sensor_manager.gyro, cycle = algorithm_manager.gyro_data_pre_process.get_gait_cycle(
            sensor_manager.gyro)
