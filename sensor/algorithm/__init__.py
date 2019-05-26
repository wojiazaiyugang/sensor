"""
算法manager
"""
from typing import Union
from enum import Enum

from sensor.algorithm.activity_recognition import ActivityRecognitionNetwork
# from sensor.algorithm.one_class_svm import AccOneClassSvm, GyroOneClassSvm
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
        self.acc_data_pre_process = AccDataPreProcess()
        self.gyro_data_pre_process = GyroDataPreProcess()
        self.ang_data_pre_process = AngDataPreProcess()

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

        self.reserved_data_count = 5  # 如果检测到了步态周期，那么用于检测的数据并不会完全清除，这样会破坏周期左边缘的检测，而是保留一定数量的点

        self.is_walking = False
        self.who_you_are = None  # 身份识别

        self.cycle_detect_history = {cycle_detect_result: 0 for cycle_detect_result in CycleDetectResult} # 周期检测的历史纪录，用于记录次数

    def _update_acc_gait_cycle(self) -> Union[np.ndarray, None]:
        self.last_acc_cycle = self.acc_data_pre_process.get_gait_cycle(self._sensor_manager.acc_to_detect_cycle)
        if self.last_acc_cycle is not None:
            self.last_validate_acc_cycle = self.last_acc_cycle
            self._sensor_manager.acc_to_detect_cycle = self._sensor_manager.acc_to_detect_cycle[
                                                       -self.reserved_data_count:]
        return self.last_acc_cycle

    def _update_gyro_gait_cycle(self) -> Union[np.ndarray, None]:
        self.last_gyro_cycle = self.gyro_data_pre_process.get_gait_cycle(self._sensor_manager.gyro_to_detect_cycle)
        if self.last_gyro_cycle is not None:
            self.last_validate_gyro_cycle = self.last_gyro_cycle
            self._sensor_manager.gyro_to_detect_cycle = self._sensor_manager.gyro_to_detect_cycle[
                                                        -self.reserved_data_count:]
        return self.last_gyro_cycle

    def _update_ang_gait_cycle(self) -> Union[np.ndarray, None]:
        self.last_ang_cycle = self.ang_data_pre_process.get_gait_cycle(self._sensor_manager.ang_to_detect_cycle)
        if self.last_ang_cycle is not None:
            self.last_validate_ang_cycle = self.last_ang_cycle
            self._sensor_manager.ang_to_detect_cycle = self._sensor_manager.ang_to_detect_cycle[
                                                       -self.reserved_data_count:]
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
        #         [np.array(self._sensor_manager.acc)[-100:, 1:]])
        #     predict_number = int(np.argmax(predict_result[0]))
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
                np.concatenate((self.last_validate_acc_cycle, self.last_validate_gyro_cycle), axis=1).T)
        else:
            return None

    def _is_walking(self) -> bool:
        """
        判断当前在行走，直接阈值判断
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
        if self.is_walking:
            # 更新步态
            self._update_acc_gait_cycle()
            self._update_gyro_gait_cycle()
            self._update_ang_gait_cycle()
            # 更新身份识别
            self.who_you_are = self.get_who_you_are()
            # 更新稳定性
            if self.last_acc_cycle is not None or self.last_gyro_cycle is not None:
                self.cycle_detect_history[CycleDetectResult.CYCLE_DETECTED] += 1
            else:
                self.cycle_detect_history[CycleDetectResult.WALK_BUT_NO_CYCLE] += 1
        else:
            self.who_you_are = ""
            self._sensor_manager.clear_data_to_detect_cycle()
            self.acc_data_pre_process.clear_template()
            self.gyro_data_pre_process.clear_template()
            self.ang_data_pre_process.clear_template()
            self.cycle_detect_history[CycleDetectResult.NOT_WALKING] += 1

        # if self.algorithm_manager.is_walking:
        #     if self.algorithm_manager.last_acc_cycle is not None or self.algorithm_manager.last_gyro_cycle is not None:
        #         self.walk_stability.append(StabilityLevel.WALK_BUT_NO_CYCLE.value[0])
        #     else:
        #         self.walk_stability.append(StabilityLevel.CYCLE_DETECTED.value[0])
        # else:
        #     self.walk_stability.append(StabilityLevel.NOT_WALKING.value[0])
