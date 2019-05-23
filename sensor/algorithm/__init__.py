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
        # 最新的一组cycle，用来确定当前是否在走路
        self.last_acc_cycle = None
        self.last_gyro_cycle = None
        self.last_ang_cycle = None
        # 最新的一组不为空的cycle，用来判断是谁
        self.last_validate_acc_cycle = None
        self.last_validate_gyro_cycle = None
        self.last_validate_ang_cycle = None

        self.is_walking = False  # 当前是否正在步行
        self.last_walk_detected_time = None # 上一次检测到步行的时间，用来维持一段时间内的步行状态
        self.walk_duration = 5000 # 如果检测到了步行，那么接下来的duration时间内都认为是在步行

    def get_acc_gait_cycle(self) -> Union[numpy.ndarray, None]:
        self._sensor_manager.acc, self.last_acc_cycle = self.acc_data_pre_process.get_gait_cycle(self._sensor_manager.acc)
        if self.last_acc_cycle is not None:
            self.last_validate_acc_cycle = self.last_acc_cycle
        return self.last_acc_cycle

    def get_gyro_gait_cycle(self) -> Union[numpy.ndarray, None]:
        self._sensor_manager.gyro, self.last_gyro_cycle = self.gyro_data_pre_process.get_gait_cycle(self._sensor_manager.gyro)
        if self.last_gyro_cycle is not None:
            self.last_validate_gyro_cycle = self.last_gyro_cycle
        return self.last_gyro_cycle

    def get_ang_gait_cycle(self) -> Union[numpy.ndarray, None]:
        self._sensor_manager.ang, self.last_ang_cycle = self.gyro_data_pre_process.get_gait_cycle(self._sensor_manager.ang)
        if self.last_ang_cycle is not None:
            self.last_validate_ang_cycle = self.last_ang_cycle
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

    def get_who_you_are(self) -> Union[int, None]:
        """
        判断当前是谁， 使用acc和gyro扔进CNN
        :return:
        """
        if self.last_validate_acc_cycle is not None and self.last_validate_gyro_cycle is not None:
            return self.cnn.get_who_you_are(numpy.concatenate((self.last_validate_acc_cycle, self.last_validate_gyro_cycle), axis=1).T)
        else:
            return None

    def is_walk_like_data0(self) -> bool:
        """
        判断当前是否像data0一样行走，利用one class svm
        :return:
        """
        return True
        # acc_predict_result = bool(self.last_acc_cycle is not None and self.acc_one_class_svm.predict(numpy.array([self.last_acc_cycle]))[0] == 1)
        # gyro_predict_result = bool(self.last_gyro_cycle is not None and self.gyro_one_class_svm.predict(numpy.array([self.last_gyro_cycle]))[0] == 1)
        # is_walking = acc_predict_result or gyro_predict_result
        # if is_walking:
        #     self.last_walk_detected_time = get_current_timestamp()
        # else:
        #     if self.last_walk_detected_time and get_current_timestamp() - self.last_walk_detected_time < self.walk_duration:
        #         is_walking = True
        # return is_walking
