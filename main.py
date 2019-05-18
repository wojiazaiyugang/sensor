import os

import cv2
import numpy
import matplotlib.pyplot as plt

from sensor.gui import GuiManager
from sensor.sensor import SensorManager
from sensor.algorithm import AlgorithmManager
from util import get_data0_data, validate_raw_data_with_timestamp
from settings import DATA_DIR, DataType

DEBUG = False

if __name__ == "__main__":
    if not DEBUG:
        GuiManager().run()
    sensor_manager = SensorManager()
    algorithm_manager = AlgorithmManager(sensor_manager)
    data = get_data0_data(os.path.join(DATA_DIR, "data0", "gyrData0.txt"))
    validate_raw_data_with_timestamp(data)
    algorithm_manager.data_pre_process.DEBUG = True
    d = []
    geis = []
    # fig = plt.figure()
    for i in range(len(data)):
        d.append(data[i])
        d, cycle = algorithm_manager._get_gait_cycle(DataType.gyro, d, gait_cycle_threshold=0.4,
                                                     expect_duration=(800, 1400))
        # if cycle is None:
        #     continue
        # plt.clf()
        # plt.plot(cycle[:, 1], color="black", linewidth=20)
        # fig.canvas.draw()
        # gei = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep="").reshape(
        #     fig.canvas.get_width_height()[::-1] + (3,))
        # geis.append(gei)
        # cv2.imshow("1", numpy.average(geis[-30:], axis=0).astype("uint8"))
        # cv2.waitKey()