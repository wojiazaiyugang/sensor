"""
算法manager
"""
from typing import Union
from enum import Enum

from sensor.algorithm.activity_recognition import ActivityRecognitionNetwork
from sensor.algorithm.one_class_svm import AccOneClassSvm, GyroOneClassSvm
from sensor.algorithm.data_pre_process import AccDataPreProcess, GyroDataPreProcess, AngDataPreProcess
from sensor.algorithm.cnn import CnnNetwork
from sensor.sensor import SensorManager
from settings import np


class CycleDetectResult(Enum):
    NOT_WALKING = "非步行",
    WALK_BUT_NO_CYCLE = "步行未稳定",
    CYCLE_DETECTED = "步行稳定",


class AlgorithmManager:
    def __init__(self, sensor_manager: SensorManager):
        self._sensor_manager = sensor_manager
        # self.activity_recognition_network = ActivityRecognitionNetwork()
        self.acc_data_pre_process = AccDataPreProcess(sensor_manager)
        self.gyro_data_pre_process = GyroDataPreProcess(sensor_manager)
        self.ang_data_pre_process = AngDataPreProcess(sensor_manager)

        self.cnn = CnnNetwork()
        # 保证one class svm在cnn下面，因为svm会使用cnn生成的数据
        self.acc_one_class_svm = AccOneClassSvm()
        self.gyro_one_class_svm = GyroOneClassSvm()
        self.is_walking = False
        self.who_you_are = None  # 身份识别

        self.cycle_detect_history = {cycle_detect_result: 0 for cycle_detect_result in
                                     CycleDetectResult}  # 周期检测的历史纪录，用于记录次数

        self.stability = []  # 步态稳定性

    def get_who_you_are(self) -> Union[int, None]:
        """
        判断当前是谁， 使用acc和gyro扔进CNN
        :return:
        """
        if self.acc_data_pre_process.validate_cycles and self.gyro_data_pre_process.validate_cycles:
            return self.cnn.get_who_you_are(
                np.concatenate(
                    (self.acc_data_pre_process.validate_cycles[-1], self.gyro_data_pre_process.validate_cycles[-1]),
                    axis=1).T)
        else:
            return None

    def _is_walking(self) -> bool:
        """
        判断当前是否在行走，直接阈值判断
        :return:
        """
        mag_interval = (20, 900)
        test_data = np.array(self._sensor_manager.acc_to_display[-100:])
        if not len(test_data):
            return False
        mag = [d[1] * d[1] + d[2] * d[2] + d[3] * d[3] for d in test_data]
        is_mag_ok = min(mag) >= mag_interval[0] and max(mag) <= mag_interval[1]
        ok_threshold = 5
        is_x_ok = max(test_data[:, 1]) - min(test_data[:, 1]) > ok_threshold
        is_y_ok = max(test_data[:, 2]) - min(test_data[:, 2]) > ok_threshold
        is_z_ok = max(test_data[:, 3]) - min(test_data[:, 3]) > ok_threshold
        is_walking = is_mag_ok and (is_x_ok or is_y_ok or is_z_ok)
        if not is_walking:  # 没在走路的话去清空数据
            self._sensor_manager.clear_data_to_detect_cycle()
        return is_walking

    def update_data(self):
        """
        更新所有算法的所有结果值
        :return:
        """
        # 更新是否在走路
        self.is_walking = self._is_walking()
        # 更新步态稳定性
        self.stability.append(self._get_stability())
        self.update_cycle_detect_result(self.is_walking)
        if self.is_walking:
            # 更新步态
            self.acc_data_pre_process.update_gait_cycle()
            self.gyro_data_pre_process.update_gait_cycle()
            self.ang_data_pre_process.update_gait_cycle()
            # 更新身份识别
            self.who_you_are = self.get_who_you_are()
        else:
            self.who_you_are = ""
            self._sensor_manager.clear_data_to_detect_cycle()
            self.acc_data_pre_process.clear_template()
            self.gyro_data_pre_process.clear_template()
            self.ang_data_pre_process.clear_template()

    def _get_stability(self) -> int:
        """
        计算一下步态的稳定性
        :return:
        """
        if not self.is_walking:
            return 0
        if self.acc_data_pre_process.last_cycle is None:
            return 1
        return 2

    def update_cycle_detect_result(self, is_walking):
        """
        更新步态检测结果，用于在GUI上显示历史记录
        :param is_walking:
        :return:
        """
        if self.acc_data_pre_process.last_cycle is not None:
            print(self.acc_one_class_svm.predict(np.array([self.acc_data_pre_process.last_cycle])))
        if is_walking:
            if self.acc_data_pre_process.last_cycle is not None or self.gyro_data_pre_process.last_cycle is not None:
                self.cycle_detect_history[CycleDetectResult.CYCLE_DETECTED] += 1
            else:
                self.cycle_detect_history[CycleDetectResult.WALK_BUT_NO_CYCLE] += 1
        else:
            self.cycle_detect_history[CycleDetectResult.NOT_WALKING] += 1
