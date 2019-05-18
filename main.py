import os

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
    algorithm_manager.gyro_data_pre_process.DEBUG = True
    data = get_data0_data(os.path.join(DATA_DIR, "data0", "gyrData0.txt"))[300:]
    d = []
    for i in range(len(data)):
        d.append(data[i])
        d, cycle = algorithm_manager.gyro_data_pre_process.get_gait_cycle(d)