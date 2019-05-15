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

from settings import DATA_DIR
from util import get_data0_data, validate_raw_data_with_timestamp, validate_raw_data_without_timestamp


class DataPreProcess:
    def __init__(self):
        self.template_duration = 1000  # 模板的长度，单位ms
        self.template = None  # 模板
        self.template_mag_threshold = 8  # 确定模板最低点的阈值，当一个点小于这个值的时候，就把这个点左右两侧共self.template_duration的窗口作为模板
        self.gait_cycle_threshold = None  # 确定最低点的阈值
        self.point_count_per_cycle = 200  # 插值的时候一个周期里点的个数

        # 把生成的步态转换为gei图像存储
        self.acc_geis = []
        self.gyro_geis = []

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

    def _find_first_gait_cycle(self, data: numpy.ndarray) -> Union[
        numpy.ndarray, None]:
        """
        检测寻找第一个步态周期
        :param template: 模板
        :param data: 原始数据
        :return: 步态周期
        """
        validate_raw_data_with_timestamp(data)
        mags = self._mag(data[:, 1:])
        if self.template is None:
            self.template = self._find_new_template(data)
        if self.template is None:
            return None
        cycle_index_points = []
        template_mag = self._mag(self.template[:, 1:])
        corr_distance = []
        for i in range(len(mags) - len(self.template) + 1):
            corr_distance.append(self._corr_distance(template_mag, mags[i:i + len(self.template)]))
            if i >= 2 and corr_distance[i - 1] < min(corr_distance[i - 2], corr_distance[i]) and corr_distance[
                i - 1] < self.gait_cycle_threshold:
                cycle_index_points.append(i - 1)
                if len(cycle_index_points) == 2:
                    cycle = data[cycle_index_points[0]:cycle_index_points[1] + 1]
                    self.template = self._updata_template(cycle)
                    return cycle
        self.template = None
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
            if 0 < index < len(mags) - 1 and mags[index] < min(mags[index - 1], mags[index + 1]) and mags[index] < self.template_mag_threshold:
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
        :param data:检测到了周期就返回 (a_mag, a_n1, a_n2, a_n3)，n1表示new axis 1。否则返回None
        :return:
        """
        assert data_type in ["acc", "gyro"], "data type 错误"
        if data_type == "acc":
            self.gait_cycle_threshold = 0.2
        elif data_type == "gyro":
            self.gait_cycle_threshold = 0.2
        if not data.any():
            return None
        validate_raw_data_with_timestamp(data)
        cycle = self._find_first_gait_cycle(data)
        if cycle is None:
            return None
        transformed_cycle = self._transform(cycle)
        if len(transformed_cycle) < 4:  # 点的数量太少无法插值
            return None
        interpolated_cycle = self._interpolate(transformed_cycle)
        return interpolated_cycle

    def _interpolate(self, data: numpy.ndarray) -> numpy.ndarray:
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

    def _transform(self, matrix_a: numpy.ndarray) -> numpy.ndarray:
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
        eigenvalue, eigenvector = numpy.linalg.eig(sigma)
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
        if self.template is None:
            return cycle
        if len(cycle) < len(self.template):
            return self.template
        return 0.9 * self.template + 0.1 * cycle[:len(self.template)]


if __name__ == "__main__":
    data = get_data0_data(os.path.join(DATA_DIR, "data0", "accData6.txt"))
    g = DataPreProcess()
    d = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    for i in range(len(data)):
        d.append(data[i])
        result = g.pre_process(numpy.array(d))
        if result is not None:
            d = []
            ax.cla()
            ax.plot(result[:, 1], color="black", linewidth=20)
            fig.canvas.draw()
            gei = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep="").reshape(
                fig.canvas.get_width_height()[::-1] + (3,))
            g.acc_geis.append(gei)
            print(len(g.acc_geis))
            cv2.imshow("1", numpy.average(g.acc_geis[-30:], axis=0).astype("uint8"))
            cv2.waitKey(30)
        if len(d) > 400:
            d = []
