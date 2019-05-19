"""
算法manager
"""
from typing import Union, Tuple

import numpy
import matplotlib.pyplot as plt
# from sensor.algorithm.activity_recognition import ActivityRecognitionNetwork
# from sensor.algorithm.svm.one_class_svm import OneClassSvm
from sensor.algorithm.data_pre_process import AccDataPreProcess, GyoDataPreProcess
from sensor.sensor import SensorManager
from util import get_current_timestamp, validate_raw_data_with_timestamp
from settings import logger, DataType


class AlgorithmManager:
    def __init__(self, sensor_manager: SensorManager):
        self._sensor_manager = sensor_manager
        # self.activity_recognition_network = ActivityRecognitionNetwork()
        # self.one_class_svm_on_data0 = OneClassSvm()
        self.acc_data_pre_process = AccDataPreProcess()
        self.gyro_data_pre_process = GyoDataPreProcess()

        self.count_threshold_clear = 400  # 阈值，超过这个阈值还没有生成步态就认为数据有问题，直接清除数据
        self.fig = plt.plot([], [])

    def get_acc_gait_cycle(self) -> Union[numpy.ndarray, None]:
        new_list, cycle = self.acc_data_pre_process.get_gait_cycle(self._sensor_manager.acc)
        self._sensor_manager.acc = new_list
        return cycle

    def get_gyro_gait_cycle(self) -> Union[numpy.ndarray, None]:
        new_list, cycle = self.gyro_data_pre_process.get_gait_cycle(self._sensor_manager.gyro)
        self._sensor_manager.gyro = new_list
        return cycle
    
    def get_ang_gait_cycle(self) -> Union[numpy.ndarray, None]:
        new_list, cycle = self.gyro_data_pre_process.get_gait_cycle(self._sensor_manager.ang)
        self._sensor_manager.ang = new_list
        return cycle

    def get_current_activity(self) -> int:
        """
        获取当前的运动状态
        :return:
        """
        return -1
        # 预测动作
        # TODO 预测一个数据至少需要100个点，100现在是写死的
        # if len(self._sensor_manager.acc) >= 100:
        #     predict_result = self.activity_recognition_network.predict(
        #         [numpy.array(self._sensor_manager.acc)[-100:, 1:]])
        #     predict_number = int(numpy.argmax(predict_result[0]))
        #     return predict_number
        # else:
        #     return -1

    def is_walk_like_data0(self) -> bool:
        """
        判断当前是否像data0一样行走，利用one class svm
        :return:
        """
        return False
        # current_timestamp = get_current_timestamp()
        # p = len(self._sensor_manager.acc) - 1
        # while p >= 0 and current_timestamp - self._sensor_manager.acc[p][0] < self.one_class_svm_on_data0.duration:
        #     p -= 1
        # if p >= 0:
        #     return self.one_class_svm_on_data0.predict_acc(numpy.array(self._sensor_manager.acc)[p:, 1:])
        # else:
        #     return False
