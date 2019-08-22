"""
步态周期相关
"""
from typing import Tuple, Union

import math
import matplotlib
import fastdtw
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate

from settings import logger, np
from util import validate_raw_data_with_timestamp, validate_raw_data_without_timestamp
from sensor.sensor import SensorManager




class DataPreProcess:
    def __init__(self, sensor_manager: SensorManager):
        self.sensor_manager = sensor_manager
        self.template_duration = 1000  # 模板的长度，单位ms
        self.gait_cycle_threshold = self.gait_cycle_threshold
        self.template = None  # 模板
        self.last_cycle = None
        self.validate_cycles = []
        self.last_cycle_to_locate = None  # 上一个周期 用于防止周期偏移
        self.cycle_duration = 0  # 上一个周期的时长
        self.data_type = self.data_type
        self.count_threshold_to_clear_template = 400  # 用了这么多的点的数据都没有找到步态周期，估计是模板有问题，清除
        self.point_count_per_cycle = 200  # 插值的时候一个周期里点的个数
        self.expect_gait_cycle_duration = self.expect_gait_cycle_duration  # 步态周期的阈值，如果检测出来的步态周期的时间不在这个范围内，就认为检测出来的是有问题的，不使用
        self.reserved_data_count = 5  # 如果检测到了步态周期，那么用于检测的数据并不会完全清除，这样会破坏周期左边缘的检测，而是保留一定数量的点
        self.DEBUG = None  # 用于显示debug信息
        self.cycle_count = 0

    def _lowpass(self, data: np.ndarray) -> np.ndarray:
        """
        低通滤波
        :param data:
        :return:
        """
        b, a = signal.butter(8, 0.2, "lowpass")
        try:
            return signal.filtfilt(b, a, data)
        except ValueError:
            return data

    def _mag(self, data: np.ndarray) -> np.ndarray:
        """
        计算合加速度
        :param data:
        :return:
        """
        validate_raw_data_without_timestamp(data)
        result = np.array([math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) for d in data])
        # return self.lowpass(result)
        return result

    def _corr_distance(self, list1: np.ndarray, list2: np.ndarray) -> float:
        """
        计算两个向量的相关距离
        :param list1:
        :param list2:
        :return:
        """
        assert len(list1) == len(list2), "比较距离时两个向量长度应该相等"
        list1 = list1 - np.average(list1)
        list2 = list2 - np.average(list2)
        return 1 - sum(list1 * list2) / (np.linalg.norm(list1) * np.linalg.norm(list2))

    def fast_dtw(self, a: np.ndarray, b: np.ndarray) -> float:
        distance, _ = fastdtw.fastdtw(a, b)
        return distance

    def _find_first_gait_cycle(self, data: np.ndarray) -> Union[np.ndarray, None]:
        """
        检测寻找第一个步态周期
        :param data: 原始数据
        :return: 步态周期
        """
        validate_raw_data_with_timestamp(data)
        mags = self._mag(data[:, 1:])
        if self.template is None:
            self.template = self._find_new_template(data)
            # print("template", self.template)
            # if self.template is not None:
            #     print(len(self.template))
        if self.template is None:
            return None
        cycle_index_points = []
        corr_distance = []
        for i in range(len(mags) - len(self.template) + 1):
            corr_distance.append(self._corr_distance(self.template, mags[i:i + len(self.template)]))
        corr_distance = self._lowpass(np.array(corr_distance))
        for i in range(len(corr_distance)):
            if i >= 2 and corr_distance[i - 1] < min(corr_distance[i - 2], corr_distance[i]) and \
                    corr_distance[i - 1] < self.gait_cycle_threshold:
                cycle_index_points.append(i - 1)
                if len(cycle_index_points) == 2:
                    # 如果找到的周期时间不够的话，就凑上下一个周期
                    self.cycle_duration = int(data[cycle_index_points[1]][0]) - int(data[cycle_index_points[0]][0])
                    if self.cycle_duration < self.expect_gait_cycle_duration[0]:
                        del cycle_index_points[-1]
                        continue
                    elif self.cycle_duration > self.expect_gait_cycle_duration[1]:
                        cycle_index_points[0] = cycle_index_points[1]
                        del cycle_index_points[-1]
                        continue
                if len(cycle_index_points) == 3:
                    self.cycle_duration = int(data[cycle_index_points[2]][0]) - int(data[cycle_index_points[1]][0])
                    if self.cycle_duration < self.expect_gait_cycle_duration[0]:
                        del cycle_index_points[-1]
                        continue
                    elif self.cycle_duration > self.expect_gait_cycle_duration[1]:
                        cycle_index_points[0] = cycle_index_points[2]
                        del cycle_index_points[-1]
                        del cycle_index_points[-1]
                        continue
                if len(cycle_index_points) == 4:
                    self.cycle_duration = int(data[cycle_index_points[3]][0]) - int(data[cycle_index_points[2]][0])
                    if self.cycle_duration < self.expect_gait_cycle_duration[0]:
                        del cycle_index_points[-1]
                        continue
                    elif self.cycle_duration > self.expect_gait_cycle_duration[1]:
                        cycle_index_points[0] = cycle_index_points[3]
                        del cycle_index_points[-1]
                        del cycle_index_points[-1]
                        del cycle_index_points[-1]
                        continue
                    if self.last_cycle_to_locate is None:
                        cycle = data[cycle_index_points[0]:cycle_index_points[2] + 1]
                        self.last_cycle_to_locate = cycle
                        if self.DEBUG:
                            use_first_cycle = True
                    else:
                        cycle1 = data[cycle_index_points[0]:cycle_index_points[2] + 1]
                        cycle2 = data[cycle_index_points[1]:cycle_index_points[3] + 1]
                        if self.fast_dtw(self.last_cycle_to_locate[:, 3], cycle1[:, 3]) < self.fast_dtw(
                                self.last_cycle_to_locate[:, 3], cycle2[:, 3]):  # 使用4格 + z轴fastdtw来寻找周期
                            cycle = cycle1
                            if self.DEBUG:
                                use_first_cycle = True
                        else:
                            cycle = cycle2
                            if self.DEBUG:
                                use_first_cycle = False
                        self.last_cycle_to_locate = self._update_last_cycle(cycle)
                    if self.DEBUG:
                        plt.cla()
                        plt.plot(data[:, 1], "r", label="x")
                        plt.plot(data[:, 2], "g", label="y")
                        plt.plot(data[:, 3], "b", label="z")
                        plt.plot(self._mag(data[:, 1:]), "black", label="mag")
                        plt.plot(corr_distance, "y", label="dis")
                        plt.axvline(cycle_index_points[0], color="r" if use_first_cycle else "b")
                        plt.axvline(cycle_index_points[1], color="r" if not use_first_cycle else "b")
                        plt.axvline(cycle_index_points[2], color="r" if use_first_cycle else "b")
                        plt.axvline(cycle_index_points[3], color="r" if not use_first_cycle else "b")
                        plt.legend()
                        plt.title("search cycle")
                        plt.show()
                    self.template = self._update_template(cycle)
                    return cycle
        return None

    def get_cycle_feature_for_gui(self) -> str:
        """
        计算出步态周期之后，计算周期的若干特征，显示在GUI上
        :return:
        """
        gait_cycle_feature_text_template = "{0} \n" \
                                           "{1}{2}{3}\n" \
                                           "{4}{5}{6}\n" \
                                           "{7}{8}{9}"
        if self.last_cycle is not None:
            return gait_cycle_feature_text_template.format(
                "【周期时长：{0:<10}】".format(self.cycle_duration * 2),
                "【X平均值:{0:7.2f}】".format(self.last_cycle[:, 1].mean(), 2),
                "【Y平均值:{0:7.2f}】".format(self.last_cycle[:, 2].mean(), 2),
                "【Z平均值:{0:7.2f}】".format(self.last_cycle[:, 3].mean(), 2),
                "【X最大值:{0:7.2f}】".format(self.last_cycle[:, 1].max(), 2),
                "【Y最大值:{0:7.2f}】".format(self.last_cycle[:, 2].max(), 2),
                "【Z最大值:{0:7.2f}】".format(self.last_cycle[:, 3].max(), 2),
                "【X标准差:{0:7.2f}】".format(self.last_cycle[:, 1].std(ddof=1), 2),
                "【Y标准差:{0:7.2f}】".format(self.last_cycle[:, 2].std(ddof=1), 2),
                "【Z标准差:{0:7.2f}】".format(self.last_cycle[:, 3].std(ddof=1), 2))
        return gait_cycle_feature_text_template.format(*[" " * 40 for _ in range(10)])

    def _find_new_template(self, data) -> Union[np.ndarray, None]:
        """
        初始化的时候或者找不到步态周期的时候需要重新寻找模板
        step1:寻找第一个局部最小点
        step2:在局部最小点周围1S内寻找最小点
        step3:最小点周围1S作为模板
        :param data:
        :return:
        """
        validate_raw_data_with_timestamp(data)
        mags = np.array([math.sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3]) for d in data])
        for index in range(len(mags)):
            # step1
            if 0 < index < len(mags) - 1 and mags[index] < min(mags[index - 1], mags[index + 1]):
                start, end = index, index
                while start >= 0 and end < len(mags) and data[end][0] - data[start][0] < self.template_duration:
                    start -= 1
                    end += 1
                # step2
                if start < 0 or end >= len(mags):
                    continue
                window_around_local_minimum = mags[start: end]
                # step3
                minimum_point_index = start + np.argmin(window_around_local_minimum)
                start, end = minimum_point_index, minimum_point_index
                while start >= 0 and end < len(data) and data[end][0] - data[start][0] < self.template_duration:
                    start -= 1
                    end += 1
                if start < 0 or end >= len(data):
                    continue
                # TODO 实时数据acc有问题是因为这里模板找的太长了 100
                # print(end - start)
                # print("start end", data[start:end+1])
                # print(end - start + 1)
                return self._mag(data[start:end][:, 1:])
        return None

    def clear_template(self):
        self.template = None

    def interpolate(self, data: np.ndarray) -> np.ndarray:
        """
        对数据进行插值
        :param data: 一个list，里面是n个list，每个list里面是若干个[x,y,z]
        :return: 一个list，里面是n个list，每个list里面是:POINT_NUMBER_PER_CYCLE个插值完的[x,y,z]
        """
        validate_raw_data_with_timestamp(data)  # 这里也是有四列，不过第一列不是时间而是合成加速度，校验函数通用
        mag_old, x_old, y_old, z_old = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        x = np.linspace(0, len(data), len(data))
        x_index = np.linspace(0, len(data), self.point_count_per_cycle)
        new_mag = interpolate.interp1d(x, mag_old, kind="quadratic")(x_index)
        new_x = interpolate.interp1d(x, x_old, kind="quadratic")(x_index)
        new_y = interpolate.interp1d(x, y_old, kind="quadratic")(x_index)
        new_z = interpolate.interp1d(x, z_old, kind="quadratic")(x_index)
        return np.array([new_mag, new_x, new_y, new_z]).T

    def transform(self, matrix_a: np.ndarray) -> np.ndarray:
        """
        将步态周期进行坐标转换
        :param matrix_a: 周期
        :return:
        """
        validate_raw_data_with_timestamp(matrix_a)
        matrix_a = matrix_a[:, 1:]
        vector_p_k = np.average(matrix_a, axis=0).T
        vector_n1 = vector_p_k / np.linalg.norm(vector_p_k)  # 一撇撇
        vector_a_n1 = np.dot(matrix_a, vector_n1)
        matrix_a_f = matrix_a - np.dot(vector_a_n1[:, np.newaxis], vector_n1[np.newaxis, :])
        u = np.average(matrix_a_f, axis=0)
        matrix_a_norm_f = matrix_a_f - u
        sigma = np.dot(matrix_a_norm_f.T, matrix_a_norm_f) / (matrix_a.shape[0] - 1)
        eigenvalue, eigenvector = np.linalg.eig(sigma)
        vector_n2 = eigenvector[np.argmax(eigenvalue)]  # 两撇撇
        vector_n3 = np.cross(vector_n1, vector_n2)
        vector_a_n2 = np.dot(matrix_a, vector_n2)
        vector_a_n3 = np.dot(matrix_a, vector_n3)
        return np.array([self._mag(matrix_a), vector_a_n1, vector_a_n2, vector_a_n3]).T

    def _update_template(self, cycle: np.ndarray) -> np.ndarray:
        """
        更新模板
        :param cycle:
        :return: 更新后的模板
        """
        if len(cycle) < len(self.template):
            return self.template
        return 0.8 * self.template + 0.2 * self._mag(cycle[:len(self.template)][:, 1:])

    def _update_last_cycle(self, cycle: np.ndarray) -> np.ndarray:
        """
        更新last cycle
        :param cycle:
        :return:
        """
        if len(cycle) < len(self.last_cycle_to_locate):
            return self.last_cycle_to_locate
        return 0.5 * self.last_cycle_to_locate + 0.5 * cycle[:len(self.last_cycle_to_locate)]

    def get_gait_cycle(self, data: list) -> Union[np.ndarray, None]:
        """
        获取步态周期
        :return: 步态周期
        """
        if len(data) == 0:
            return None
        validate_raw_data_with_timestamp(np.array(data))
        first_cycle = self._find_first_gait_cycle(np.array(data))
        if first_cycle is None:
            # TODO count_threshold_to_clear_template 必须保证比sensor中存的最大的点数要小，不然一直无法清除模板，这里暂时都是400，没有限制
            # print(len(data))
            if len(data) >= self.count_threshold_to_clear_template:
                self.template = None
            return None
        # transformed_cycle = self.transform(first_cycle)
        # if len(transformed_cycle) < 4:  # 点的数量太少无法插值
        #     return [], None
        # interpolated_cycle = self.interpolate(transformed_cycle)
        interpolated_cycle_without_transform = self.interpolate(first_cycle)
        self.cycle_count += 1
        if self.DEBUG:
            plt.plot(interpolated_cycle_without_transform[:, 1], "r", label="x")
            plt.plot(interpolated_cycle_without_transform[:, 2], "g", label="y")
            plt.plot(interpolated_cycle_without_transform[:, 3], "b", label="z")
            plt.title("interpolated_cycle")
            plt.legend()
            plt.show()
        if self.DEBUG:
            logger.debug("{0}:CYCYLE INDEX:{1}".format(self.data_type, self.cycle_count))
        return np.concatenate((np.array([self._mag(interpolated_cycle_without_transform[:, 1:])]).T,
                               interpolated_cycle_without_transform[:, 1:]), axis=1)

    def update_gait_cycle(self) -> Union[np.ndarray, None]:
        data = self.get_data()
        self.last_cycle = self.get_gait_cycle(data)
        if self.last_cycle is not None:
            self.validate_cycles.append(self.last_cycle)
            self.update_data_to_detect()
        return self.last_cycle

    def get_data(self) -> list:
        """
        获取用于检测周期的原始数据，从data_to_detect中获得
        :return:
        """
        raise NotImplementedError

    def update_data_to_detect(self):
        """
        更新data_to_detect
        :return:
        """
        raise NotImplementedError


class AccDataPreProcess(DataPreProcess):

    def __init__(self, sensor_manager: SensorManager):
        self.data_type = "加速度"
        self.gait_cycle_threshold = 1
        self.expect_gait_cycle_duration = (400, 700)
        super().__init__(sensor_manager)

    def get_data(self):
        return self.sensor_manager.acc_to_detect_cycle

    def update_data_to_detect(self):
        self.sensor_manager.acc_to_detect_cycle = self.sensor_manager.acc_to_detect_cycle[-self.reserved_data_count:]


class GyroDataPreProcess(DataPreProcess):

    def __init__(self, sensor_manager: SensorManager):
        self.data_type = "陀螺仪"
        self.gait_cycle_threshold = 1
        self.expect_gait_cycle_duration = (400, 700)
        super().__init__(sensor_manager)

    def get_data(self):
        return self.sensor_manager.gyro_to_detect_cycle

    def update_data_to_detect(self):
        self.sensor_manager.gyro_to_detect_cycle = self.sensor_manager.gyro_to_detect_cycle[-self.reserved_data_count:]
