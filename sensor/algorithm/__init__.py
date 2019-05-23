"""
算法manager
"""
from typing import Union, Tuple

import numpy
import matplotlib.pyplot as plt
from sensor.algorithm.activity_recognition import ActivityRecognitionNetwork
from sensor.algorithm.one_class_svm import AccOneClassSvm, GyroOneClassSvm
from sensor.algorithm.data_pre_process import AccDataPreProcess, GyroDataPreProcess
from sensor.algorithm.cnn import CnnNetwork
from sensor.sensor import SensorManager
from util import get_current_timestamp, validate_raw_data_with_timestamp
from settings import logger, DataType


class AlgorithmManager:
    def __init__(self, sensor_manager: SensorManager):
        self._sensor_manager = sensor_manager
        # self.activity_recognition_network = ActivityRecognitionNetwork()
        self.acc_data_pre_process = AccDataPreProcess()
        self.gyro_data_pre_process = GyroDataPreProcess()

        self.cnn = CnnNetwork()
        # 保证one class svm在cnn下面，因为svm会使用cnn生成的数据
        self.acc_one_class_svm = AccOneClassSvm()
        self.gyro_one_class_svm = GyroOneClassSvm()

        self.last_acc_cycle = None
        self.last_gyro_cycle = None
        self.last_ang_cycle = None

    def get_acc_gait_cycle(self) -> Union[numpy.ndarray, None]:
        self._sensor_manager.acc, self.last_acc_cycle = self.acc_data_pre_process.get_gait_cycle(self._sensor_manager.acc)
        return self.last_acc_cycle

    def get_gyro_gait_cycle(self) -> Union[numpy.ndarray, None]:
        self._sensor_manager.gyro, self.last_gyro_cycle = self.gyro_data_pre_process.get_gait_cycle(self._sensor_manager.gyro)
        return self.last_gyro_cycle

    def get_ang_gait_cycle(self) -> Union[numpy.ndarray, None]:
        self._sensor_manager.ang, self.last_ang_cycle = self.gyro_data_pre_process.get_gait_cycle(self._sensor_manager.ang)
        return self.last_ang_cycle

    def get_current_activity(self) -> int:
        """
        获取当前的运动状态
        :return:
        """
        pass
        # 预测动作
        # TODO 预测一个数据至少需要100个点，100现在是写死的
        # if len(self._sensor_manager.acc) >= 100:
        #     predict_result = self.activity_recognition_network.predict(
        #         [numpy.array(self._sensor_manager.acc)[-100:, 1:]])
        #     predict_number = int(numpy.argmax(predict_result[0]))
        #     return predict_number
        # else:
        #     return -1

    def _get_last_cycle(self):
        """
        获取最近的步态周期
        :return:
        """
        if self.last_acc_cycle is not None and self.last_gyro_cycle is not None:
            return numpy.concatenate((self.last_acc_cycle, self.last_gyro_cycle), axis= 1)
        else:
            return None

    def get_who_you_are(self) -> Union[int, None]:
        """
        判断当前是谁， 使用acc和gyro扔进CNN
        :return:
        """
        last_cycle = self._get_last_cycle()
        if last_cycle is not None:
            return self.cnn.get_who_you_are(last_cycle.T)
        else:
            return None

    def is_walk_like_data0(self) -> bool:
        """
        判断当前是否像data0一样行走，利用one class svm
        :return:
        """
        acc_predict_result = bool(self.last_acc_cycle is not None and self.acc_one_class_svm.predict(numpy.array([self.last_acc_cycle]))[0] == 1)
        gyro_predict_result = bool(self.last_gyro_cycle is not None and self.gyro_one_class_svm.predict(numpy.array([self.last_gyro_cycle]))[0] == 1)
        return acc_predict_result or gyro_predict_result
