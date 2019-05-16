"""
算法manager
"""
from typing import Union, Tuple

import numpy
import matplotlib.pyplot as plt
# from sensor.algorithm.activity_recognition import ActivityRecognitionNetwork
# from sensor.algorithm.svm.one_class_svm import OneClassSvm
from sensor.algorithm.data_pre_process import DataPreProcess
from sensor.sensor import SensorManager
from util import get_current_timestamp, validate_raw_data_with_timestamp
from settings import logger


class AlgorithmManager:
    def __init__(self, sensor_manager: SensorManager):
        self._sensor_manager = sensor_manager
        # self.activity_recognition_network = ActivityRecognitionNetwork()
        # self.one_class_svm_on_data0 = OneClassSvm()
        self.data_pre_process = DataPreProcess()

        self.count_threshold_clear = 400  # 阈值，超过这个阈值还没有生成步态就认为数据有问题，直接清除数据
        self.fig = plt.plot([], [])

    def _get_gait_cycle(self, data_type, data: list, gait_cycle_threshold: float = None, expect_duration: tuple = None) -> Tuple[list, Union[numpy.ndarray, None]]:
        """
        获取步态周期
        :return: 原始数据使用之后修改成的新的list，步态周期
        """
        validate_raw_data_with_timestamp(numpy.array(data))
        first_cycle = self.data_pre_process.find_first_gait_cycle(numpy.array(data), gait_cycle_threshold)
        if first_cycle is None:
            if len(data) > self.count_threshold_clear:
                data = []
            return data, None
        cycle_duration = int(first_cycle[-1][0]) - int(first_cycle[0][0])  # 检测出来的步态周期的时长，ms
        if not expect_duration[0] <= cycle_duration <= expect_duration[1]:
            logger.debug("无效步态，数据类型:{0},期望时长:{1},实际时长{2}".format(data_type, expect_duration, cycle_duration))
            return [], None
        transformed_cycle = self.data_pre_process.transform(first_cycle)
        if len(transformed_cycle) < 4:  # 点的数量太少无法插值
            return [], None
        interpolated_cycle = self.data_pre_process.interpolate(transformed_cycle)
        return [], interpolated_cycle

    def get_acc_gait_cycle(self) -> Union[numpy.ndarray, None]:
        """
        获取acc的一个步态周期
        :return:
        """
        new_acc_data, cycle = self._get_gait_cycle("acc", self._sensor_manager.acc, gait_cycle_threshold=0.4, expect_duration = (800,1400))
        self._sensor_manager.acc = new_acc_data
        return cycle

    def get_gyro_gait_cycle(self) -> Union[numpy.ndarray, None]:
        new_gyro_data, cycle = self._get_gait_cycle("gyro", self._sensor_manager.gyro, gait_cycle_threshold=0.2, expect_duration = (800,1400))
        self._sensor_manager.gyro = new_gyro_data
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
