"""
算法manager
"""
from typing import Union, Tuple

import numpy
import matplotlib.pyplot as plt
# from sensor.algorithm.activity_recognition import ActivityRecognitionNetwork
# from sensor.algorithm.svm.one_class_svm import OneClassSvm
from sensor.algorithm.data_pre_process import AccDataPreProcess, GyroDataPreProcess
from sensor.algorithm.cnn import CnnNetwork
from sensor.sensor import SensorManager
from util import get_current_timestamp, validate_raw_data_with_timestamp
from settings import logger, DataType


class AlgorithmManager:
    def __init__(self, sensor_manager: SensorManager):
        self._sensor_manager = sensor_manager
        # self.activity_recognition_network = ActivityRecognitionNetwork()
        # self.one_class_svm_on_data0 = OneClassSvm()
        self.acc_data_pre_process = AccDataPreProcess()
        self.gyro_data_pre_process = GyroDataPreProcess()

        self.cnn = CnnNetwork()
        self.fig = plt.plot([], [])

        self.acc_cycles = []
        self.gyro_cycles = []
        self.ang_cycles = []

    def get_acc_gait_cycle(self) -> Union[numpy.ndarray, None]:
        new_list, cycle = self.acc_data_pre_process.get_gait_cycle(self._sensor_manager.acc)
        self._sensor_manager.acc = new_list
        if cycle is not None:
            self.acc_cycles.append(cycle)
        return cycle

    def get_gyro_gait_cycle(self) -> Union[numpy.ndarray, None]:
        new_list, cycle = self.gyro_data_pre_process.get_gait_cycle(self._sensor_manager.gyro)
        self._sensor_manager.gyro = new_list
        if cycle is not None:
            self.gyro_cycles.append(cycle)
        return cycle

    def get_ang_gait_cycle(self) -> Union[numpy.ndarray, None]:
        new_list, cycle = self.gyro_data_pre_process.get_gait_cycle(self._sensor_manager.ang)
        self._sensor_manager.ang = new_list
        if cycle is not None:
            self.ang_cycles.append(cycle)
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

    def get_who_you_are(self):
        """
        判断当前是谁， 使用acc和gyro扔进CNN
        :return:
        """
        if self.acc_cycles and self.gyro_cycles:
            data = numpy.concatenate((self.acc_cycles[-1], self.gyro_cycles[-1]), axis= 1).T
        else:
            return -1
        return self.cnn.get_who_you_are(data)

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
