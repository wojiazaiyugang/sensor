import os
import time
import pickle
from sensor.algorithm import AlgorithmManager
from sensor.gui import GuiManager
from sensor.sensor import SensorManager
from settings import DATA_DIR, logger, CYCLE_FILE_DIR, SENSOR_DATA
from util import get_data0_data

DEBUG = False


if __name__ == "__main__":
    for i in range(10):
        with open(os.path.join(CYCLE_FILE_DIR, "gyro{0}".format(i)), "rb") as file_gyro, open(os.path.join(CYCLE_FILE_DIR, "acc{0}".format(i)), "rb") as file_acc:
            data = pickle.loads(file_acc.read())
            print("acc{0}:{1}".format(i,len([_ for _ in data  if _ is not None])))
            data = pickle.loads(file_gyro.read())
            print("gyro{0}:{1}".format(i, len([_ for _ in data if _ is not None])))
    exit(0)

    if not DEBUG:
        GuiManager().run()
    sensor_manager = SensorManager()
    algorithm_manager = AlgorithmManager(sensor_manager)
    algorithm_manager.gyro_data_pre_process.DEBUG = True
    while True:
        sensor_manager.mock_real_time_data_from_data0()
        sensor_manager.gyro, cycle = algorithm_manager.gyro_data_pre_process.get_gait_cycle(
            sensor_manager.gyro)
