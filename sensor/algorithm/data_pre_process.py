"""
步态周期相关
"""
import os
import math
from typing import Tuple, Union

import cv2
import numpy
# TODO: 这两句是为了兼容mac
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate

from settings import DATA_DIR, logger, DataType
from util import get_data0_data, validate_raw_data_with_timestamp, validate_raw_data_without_timestamp


class DataPreProcess:
    def __init__(self):
        self.template_duration = 1000  # 模板的长度，单位ms
        self.acc_template = None  # 模板
        self.gyro_template = None
        self.point_count_per_cycle = 200  # 插值的时候一个周期里点的个数
        self.expect_gait_cycle_duration = (400, 1200)  # 步态周期的阈值，如果检测出来的步态周期的时间不在这个范围内，就认为检测出来的是有问题的，不使用

        self.time_duration_threshold_to_clear = 3000 # 超过多长时间没有识别到成功的步态，就认为已经找到了步态但是不合格，那么就清除所有数据
        self.data_type = None  # 当前正在处理的数据类型，acc或者是gyro，用于决定不同的阈值
        # 把生成的步态转换为gei图像存储
        self.acc_geis = []
        self.gyro_geis = []

        self.DEBUG = None # 用于显示debug信息

    def _lowpass(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        低通滤波
        :param data:
        :return:
        """
        b, a = signal.butter(8, 0.001, "lowpass")
        try:
            return signal.filtfilt(b, a, data)
        except ValueError:
            return data

    def _mag(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        计算合加速度
        :param data:
        :return:
        """
        validate_raw_data_without_timestamp(data)
        result = numpy.array([math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) for d in data])
        # return self.lowpass(result)
        return result

    def _corr_distance(self, list1: numpy.ndarray, list2: numpy.ndarray) -> float:
        """
        计算两个向量的相关距离
        :param list1:
        :param list2:
        :return:
        """
        assert len(list1) == len(list2), "比较距离时两个向量长度应该相等"
        list1 = list1 - numpy.average(list1)
        list2 = list2 - numpy.average(list2)
        return 1 - sum(list1 * list2) / (numpy.linalg.norm(list1) * numpy.linalg.norm(list2))

    def find_first_gait_cycle(self, data_type: DataType, data: numpy.ndarray, gait_cycle_threshold: float, expect_duration:tuple) -> Union[numpy.ndarray, None]:
        """
        检测寻找第一个步态周期
        :param gait_cycle_threshold: 用于找最低点的阈值
        :param data: 原始数据
        :return: 步态周期
        """
        validate_raw_data_with_timestamp(data)
        mags = self._mag(data[:, 1:])
        template = None
        if data_type == DataType.acc.name:
            template = self.acc_template if self.acc_template is not None else self._find_new_template(data)
            if template is None:
                return None
        elif data_type == DataType.gyro.name:
            template = self.gyro_template if self.gyro_template is not None else self._find_new_template(data)
            if template is None:
                return None
        else:
            raise Exception("data type 错误")
        cycle_index_points = []
        template_mag = self._mag(template[:, 1:])
        corr_distance = []
        for i in range(len(mags) - len(template) + 1):
            corr_distance.append(self._corr_distance(template_mag, mags[i:i + len(template)]))
            if i >= 2 and corr_distance[i - 1] < min(corr_distance[i - 2], corr_distance[i]) and corr_distance[
                i - 1] < gait_cycle_threshold:
                cycle_index_points.append(i - 1)
                if len(cycle_index_points) == 2:
                    # 如果找到的周期时间不够的话，就凑上下一个周期
                    cycle_duration = int(data[cycle_index_points[1]][0]) - int(data[cycle_index_points[0]][0])
                    if cycle_duration < expect_duration[0]:
                        del cycle_index_points[-1]
                        continue
                    elif cycle_duration > expect_duration[1]:
                        cycle_index_points[0] = cycle_index_points[1]
                        del cycle_index_points[-1]
                        continue
                    if self.DEBUG:
                        plt.cla()
                        plt.plot(corr_distance, "r")
                        plt.axvline(cycle_index_points[0])
                        plt.axvline(cycle_index_points[1])
                        plt.plot(data[:,1],"b")
                        plt.plot(template_mag, "g")
                        plt.show()
                    cycle = data[cycle_index_points[0]:cycle_index_points[1] + 1]
                    new_template = self._updata_template(cycle)
                    if data_type == DataType.acc.name:
                        self.acc_template = new_template
                    elif data_type == DataType.gyro.name:
                        self.gyro_template = new_template
                    return cycle
        return None

    def _find_new_template(self, data) -> Union[numpy.ndarray, None]:
        """
        初始化的时候或者找不到步态周期的时候需要重新寻找模板
        step1:寻找第一个局部最小点
        step2:在局部最小点周围1S内寻找最小点
        step3:最小点周围1S作为模板
        :param data:
        :return:
        """
        validate_raw_data_with_timestamp(data)
        mags = numpy.array([math.sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3]) for d in data])
        for index in range(len(mags)):
            # step1
            # logger.info("thres:{0}".format(self.gait_cycle_threshold))
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
                minimum_point_index = start + numpy.argmin(window_around_local_minimum)
                start, end = minimum_point_index, minimum_point_index
                while start >= 0 and end < len(data) and data[end][0] - data[start][0] < self.template_duration:
                    start -= 1
                    end += 1
                if start < 0 or end >= len(data):
                    continue
                return data[start:end]
        return None

    def pre_process(self, data: numpy.ndarray, data_type: str) -> Union[numpy.ndarray, None]:
        """
        数据预处理
        1、周期检测
        2、坐标转换
        3、插值
        :param data_type: 检测的数据类型
        :param data:检测到了周期就返回 (a_mag, a_n1, a_n2, a_n3)，n1表示new axis 1。没有检测到返回None
        :return:
        """
        assert data_type in ["acc", "gyro"], "data type 错误"
        self.data_type = data_type
        if not data.any():
            return None
        validate_raw_data_with_timestamp(data)
        cycle = self.find_first_gait_cycle(data)
        if cycle is None:
            return None
        transformed_cycle = self.transform(cycle)
        if len(transformed_cycle) < 4:  # 点的数量太少无法插值
            return None
        interpolated_cycle = self.interpolate(transformed_cycle)
        return interpolated_cycle

    def interpolate(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        对数据进行插值
        :param point_number: 插值之后每个周期内的数据点个数
        :param data: 一个list，里面是n个list，每个list里面是若干个[x,y,z]
        :return: 一个list，里面是n个list，每个list里面是:POINT_NUMBER_PER_CYCLE个插值完的[x,y,z]
        """
        validate_raw_data_with_timestamp(data)  # 这里也是有四列，不过第一列不是时间而是合成加速度，校验函数通用
        mag_old, x_old, y_old, z_old = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        x = numpy.linspace(0, len(data), len(data))
        x_index = numpy.linspace(0, len(data), self.point_count_per_cycle)
        new_mag = interpolate.interp1d(x, mag_old, kind="quadratic")(x_index)
        new_x = interpolate.interp1d(x, x_old, kind="quadratic")(x_index)
        new_y = interpolate.interp1d(x, y_old, kind="quadratic")(x_index)
        new_z = interpolate.interp1d(x, z_old, kind="quadratic")(x_index)
        return numpy.array([new_mag, new_x, new_y, new_z]).T

    def transform(self, matrix_a: numpy.ndarray) -> numpy.ndarray:
        """
        将步态周期进行坐标转换
        :param matrix_a: 周期
        :return:
        """
        validate_raw_data_with_timestamp(matrix_a)
        matrix_a = matrix_a[:, 1:]
        vector_p_k = numpy.average(matrix_a, axis=0).T
        vector_n1 = vector_p_k / numpy.linalg.norm(vector_p_k)  # 一撇撇
        vector_a_n1 = numpy.dot(matrix_a, vector_n1)
        matrix_a_f = matrix_a - numpy.dot(vector_a_n1[:, numpy.newaxis], vector_n1[numpy.newaxis, :])
        u = numpy.average(matrix_a_f, axis=0)
        matrix_a_norm_f = matrix_a_f - u
        sigma = numpy.dot(matrix_a_norm_f.T, matrix_a_norm_f) / (matrix_a.shape[0] - 1)
        try:
            eigenvalue, eigenvector = numpy.linalg.eig(sigma)
        except Exception as err:
            pass
        vector_n2 = eigenvector[numpy.argmax(eigenvalue)]  # 两撇撇
        vector_n3 = numpy.cross(vector_n1, vector_n2)
        vector_a_n2 = numpy.dot(matrix_a, vector_n2)
        vector_a_n3 = numpy.dot(matrix_a, vector_n3)
        return numpy.array([self._mag(matrix_a), vector_a_n1, vector_a_n2, vector_a_n3]).T

    def _updata_template(self, cycle: numpy.ndarray) -> numpy.ndarray:
        """
        更新模板
        :param cycle:
        :return: 更新后的模板
        """
        if self.acc_template is None:
            return cycle
        if len(cycle) < len(self.acc_template):
            return self.acc_template
        return 0.9 * self.acc_template + 0.1 * cycle[:len(self.acc_template)]

    def validate_cycle(self, cycle: numpy.ndarray) -> Union[numpy.ndarray, None]:
        """
        对检测出来的步态进行校验，通过校验的才认为是一个合法的步态，否则就返回None
        :param cycle:
        :return:
        """
        validate_raw_data_with_timestamp(cycle)
        cycle = self.validate_cycle_duration(cycle)
        return cycle

    def validate_cycle_duration(self, cycle: numpy.ndarray) -> Union[numpy.ndarray, None]:
        """
        获取步态周期的有效时长
        :return:
        """
        if cycle is None:
            return None
        if self.data_type == "acc":
            expect_duration = (800, 1400)
        elif self.data_type == "gyro":
            expect_duration = (800, 1400)
        else:
            raise Exception("data type 错误")
        cycle_duration = int(cycle[-1][0]) - int(cycle[0][0])  # 检测出来的步态周期的时长，ms
        if not expect_duration[0] <= cycle_duration <= expect_duration[1]:
            logger.debug("无效步态，数据类型:{0},期望时长:{1},实际时长{2}".format(self.data_type, expect_duration,cycle_duration))
            return None
        return cycle

    def _get_gait_cycle_threshold(self) -> float:
        """
        获取阈值
        :return:
        """
        if self.data_type == "acc":
            return 0.4
        elif self.data_type == "gyro":
            return 0.2
        else:
            raise Exception("data type 错误")
