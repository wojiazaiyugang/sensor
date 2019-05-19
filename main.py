import os
import time
from sensor.algorithm import AlgorithmManager
from sensor.gui import GuiManager
from sensor.sensor import SensorManager
from settings import DATA_DIR
from util import get_data0_data

DEBUG = False

if __name__ == "__main__":
    if not DEBUG:
        GuiManager().run()
    sensor_manager = SensorManager()
    algorithm_manager = AlgorithmManager(sensor_manager)
    algorithm_manager.acc_data_pre_process.DEBUG = True
    while True:
        sensor_manager.mock_real_time_data_from_data0()
        sensor_manager.acc, cycle = algorithm_manager.acc_data_pre_process.get_gait_cycle(sensor_manager.acc)