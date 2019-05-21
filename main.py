import os
import time
import pickle
from sensor.algorithm import AlgorithmManager
from sensor.gui import GuiManager
from sensor.sensor import SensorManager
from settings import DATA_DIR, logger, CYCLE_FILE_DIR
from util import get_data0_data

DEBUG = False


if __name__ == "__main__":
    if not DEBUG:
        GuiManager().run()
    sensor_manager = SensorManager()
    algorithm_manager = AlgorithmManager(sensor_manager)
    algorithm_manager.gyro_data_pre_process.DEBUG = True
    while True:
        sensor_manager._mock_real_time_data_from_data0()
        sensor_manager.gyro, cycle = algorithm_manager.gyro_data_pre_process.get_gait_cycle(
            sensor_manager.gyro)



    # @staticmethod
    # def convert_data0_to_gait_cycles_file():
    #     """
    #     把data0的数据进行周期分割，然后存进文件里供后续CNN等使用
    #     :param people_index:
    #     :return:
    #     """
    #     if not os.path.isdir(CYCLE_FILE_DIR):
    #         os.makedirs(CYCLE_FILE_DIR)
    #
    #     acc_cycles = []
    #     gyro_cycles = []
    #     for i in range(10):
    #         sensor_manager = SensorManager(i)
    #         algorithm_manager = AlgorithmManager(sensor_manager)
    #         while True:
    #             mock_result = sensor_manager.mock_real_time_data_from_data0()
    #             if not mock_result:
    #                 break
    #             sensor_manager.acc, acc_cycle = algorithm_manager.acc_data_pre_process.get_gait_cycle(
    #                 sensor_manager.acc)
    #             sensor_manager.gyro, gyro_cycle = algorithm_manager.gyro_data_pre_process.get_gait_cycle(
    #                 sensor_manager.gyro)
    #             acc_cycles.append(acc_cycle)
    #             gyro_cycles.append(gyro_cycle)
    #         with open(os.path.join(CYCLE_FILE_DIR, "gyro{0}".format(i)), "wb") as file_gyro, \
    #                 open(os.path.join(CYCLE_FILE_DIR, "acc{0}".format(i)), "wb") as file_acc:
    #             file_acc.write(pickle.dumps(acc_cycles))
    #             file_gyro.write(pickle.dumps(gyro_cycles))
    #         logger.info("处理完毕")
