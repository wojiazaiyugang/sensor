"""
算法manager
"""
from typing import Union

import numpy
from sensor.algorithm.activity_recognition import ActivityRecognitionNetwork
# from sensor.algorithm.one_class_svm import AccOneClassSvm, GyroOneClassSvm
from sensor.algorithm.data_pre_process import AccDataPreProcess, GyroDataPreProcess
from sensor.algorithm.cnn import CnnNetwork
from sensor.sensor import SensorManager


class AlgorithmManager:
    def __init__(self, sensor_manager: SensorManager):
        self._sensor_manager = sensor_manager
        # self.activity_recognition_network = ActivityRecognitionNetwork()
        self.acc_data_pre_process = AccDataPreProcess()
        self.gyro_data_pre_process = GyroDataPreProcess()

        self.cnn = CnnNetwork()
        # 保证one class svm在cnn下面，因为svm会使用cnn生成的数据
        # self.acc_one_class_svm = AccOneClassSvm()
        # self.gyro_one_class_svm = GyroOneClassSvm()
        # 最新的一组cycle，用来确定当前是否在走路
        self.last_acc_cycle = None
        self.last_gyro_cycle = None
        self.last_ang_cycle = None
        # 最新的一组不为空的cycle，用来判断是谁
        self.last_validate_acc_cycle = None
        self.last_validate_gyro_cycle = None
        self.last_validate_ang_cycle = None

        self.reserved_data_count = 5 # 如果检测到了步态周期，那么用于检测的数据并不会完全清除，这样会破坏周期左边缘的检测，而是保留一定数量的点

        self.is_walking = False
        self.who_you_are = None # 身份识别

    def _update_acc_gait_cycle(self) -> Union[numpy.ndarray, None]:
        self.last_acc_cycle = self.acc_data_pre_process.get_gait_cycle(self._sensor_manager.acc_to_detect_cycle)
        if self.last_acc_cycle is not None:
            self.last_validate_acc_cycle = self.last_acc_cycle
            self._sensor_manager.acc_to_detect_cycle = self._sensor_manager.acc_to_detect_cycle[-self.reserved_data_count:]
        return self.last_acc_cycle

    def _update_gyro_gait_cycle(self) -> Union[numpy.ndarray, None]:
        self.last_gyro_cycle = self.gyro_data_pre_process.get_gait_cycle(self._sensor_manager.gyro_to_detect_cycle)
        if self.last_gyro_cycle is not None:
            self.last_validate_gyro_cycle = self.last_gyro_cycle
            self._sensor_manager.gyro_to_detect_cycle = self._sensor_manager.gyro_to_detect_cycle[-self.reserved_data_count:]
        return self.last_gyro_cycle

    def _update_ang_gait_cycle(self) -> Union[numpy.ndarray, None]:
        self.last_ang_cycle = self.gyro_data_pre_process.get_gait_cycle(self._sensor_manager.ang_to_detect_cycle)
        if self.last_ang_cycle is not None:
            self.last_validate_ang_cycle = self.last_ang_cycle
            self._sensor_manager.ang_to_detect_cycle = self._sensor_manager.ang_to_detect_cycle[-self.reserved_data_count:]
        return self.last_ang_cycle

    def get_current_activity(self) -> int:
        """
        获取当前的运动状态
        :return:
        """
        pass
        # 预测动作
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
            return self.cnn.get_who_you_are(
                numpy.concatenate((self.last_validate_acc_cycle, self.last_validate_gyro_cycle), axis=1).T)
        else:
            return None

    def _is_walking(self) -> bool:
        """
        判断当前在行走，直接阈值判断
        :return:
        """
        mag_interval = (20, 300)
        test_data = self._sensor_manager.acc_to_display
        if not test_data:
            return False
        mag = [d[1] * d[1] + d[2] * d[2] + d[3] * d[3] for d in test_data]
        is_walking = min(mag) >= mag_interval[0] and max(mag) <= mag_interval[1]
        if not is_walking: # 没在走路的话去清空数据
            self._sensor_manager.clear_data_to_detect_cycle()
        return is_walking

    def update_data(self):
        """
        更新所有算法的所有结果值
        :return:
        """
        # 更新是否在走路
        self.is_walking = self._is_walking()
        # 更新步态
        self._update_acc_gait_cycle()
        self._update_gyro_gait_cycle()
        self._update_ang_gait_cycle()

        self.who_you_are = self.get_who_you_are()


