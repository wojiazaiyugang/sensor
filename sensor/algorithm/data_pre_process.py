"""
步态周期相关
"""
from typing import Tuple, Union

import math
# TODO: 这两句是为了兼容mac
import matplotlib
import numpy
import fastdtw

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate

from settings import logger
from util import validate_raw_data_with_timestamp, validate_raw_data_without_timestamp


class DataPreProcess:
    def __init__(self):
        self.template_duration = 1000  # 模板的长度，单位ms
        self.gait_cycle_threshold = self.gait_cycle_threshold
        self.template = None  # 模板
        self.last_cycle = None  # 上一个周期 用于防止周期偏移
        self.data_type = self.data_type
        self.count_threshold_clear = 400
        self.point_count_per_cycle = 200  # 插值的时候一个周期里点的个数
        self.expect_gait_cycle_duration = self.expect_gait_cycle_duration  # 步态周期的阈值，如果检测出来的步态周期的时间不在这个范围内，就认为检测出来的是有问题的，不使用
        self.time_duration_threshold_to_clear = 3000  # 超过多长时间没有识别到成功的步态，就认为已经找到了步态但是不合格，那么就清除所有数据

        self.DEBUG = None  # 用于显示debug信息
        self.cycle_count = 0

    def _lowpass(self, data: numpy.ndarray) -> numpy.ndarray:
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

    def fast_dtw(self, a: numpy.ndarray, b: numpy.ndarray) -> float:
        distance, _ = fastdtw.fastdtw(a, b)
        return distance

    def _find_first_gait_cycle(self, data: numpy.ndarray) -> Union[numpy.ndarray, None]:
        """
        检测寻找第一个步态周期
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
        corr_distance = self._lowpass(numpy.array(corr_distance))
        for i in range(len(corr_distance)):
            if i >= 2 and corr_distance[i - 1] < min(corr_distance[i - 2], corr_distance[i]) and corr_distance[i - 1] < self.gait_cycle_threshold:
                cycle_index_points.append(i - 1)
                if len(cycle_index_points) == 2:
                    # 如果找到的周期时间不够的话，就凑上下一个周期
                    cycle_duration = int(data[cycle_index_points[1]][0]) - int(data[cycle_index_points[0]][0])
                    if cycle_duration < self.expect_gait_cycle_duration[0]:
                        # logger.debug("cycle 错误1:{0}".format(cycle_duration))
                        del cycle_index_points[-1]
                        continue
                    elif cycle_duration > self.expect_gait_cycle_duration[1]:
                        # logger.debug("cycle 错误2:{0}".format(cycle_duration))
                        cycle_index_points[0] = cycle_index_points[1]
                        del cycle_index_points[-1]
                        continue
                if len(cycle_index_points) == 3:
                    cycle_duration = int(data[cycle_index_points[2]][0]) - int(data[cycle_index_points[1]][0])
                    if cycle_duration < self.expect_gait_cycle_duration[0]:
                        # logger.debug("cycle 错误3:{0}".format(cycle_duration))
                        del cycle_index_points[-1]
                        continue
                    elif cycle_duration > self.expect_gait_cycle_duration[1]:
                        # logger.debug("cycle 错误4:{0}".format(cycle_duration))
                        cycle_index_points[0] = cycle_index_points[2]
                        del cycle_index_points[-1]
                        del cycle_index_points[-1]
                        continue
                if len(cycle_index_points) == 4:
                    cycle_duration = int(data[cycle_index_points[3]][0]) - int(data[cycle_index_points[2]][0])
                    if cycle_duration < self.expect_gait_cycle_duration[0]:
                        # logger.debug("cycle 错误5:{0}".format(cycle_duration))
                        del cycle_index_points[-1]
                        continue
                    elif cycle_duration > self.expect_gait_cycle_duration[1]:
                        # logger.debug("cycle 错误6:{0}".format(cycle_duration))
                        cycle_index_points[0] = cycle_index_points[3]
                        del cycle_index_points[-1]
                        del cycle_index_points[-1]
                        del cycle_index_points[-1]
                        continue
                    if self.last_cycle is None:
                        cycle = data[cycle_index_points[0]:cycle_index_points[2] + 1]
                        self.last_cycle = cycle
                        if self.DEBUG:
                            use_first_cycle = True
                    else:
                        cycle1 = data[cycle_index_points[0]:cycle_index_points[2] + 1]
                        cycle2 = data[cycle_index_points[1]:cycle_index_points[3] + 1]
                        if self.fast_dtw(self.last_cycle[:,3], cycle1[:, 3]) < self.fast_dtw(self.last_cycle[:,3], cycle2[:, 3]): # 使用4格 + z轴fastdtw来寻找周期
                            cycle = cycle1
                            if self.DEBUG:
                                use_first_cycle = True
                        else:
                            cycle = cycle2
                            if self.DEBUG:
                                use_first_cycle = False
                        self.last_cycle = self._update_last_cycle(cycle)
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
                    self.template = self._updata_template(cycle)
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

    def interpolate(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        对数据进行插值
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
        if len(cycle) < len(self.template):
            return self.template
        return 0.8 * self.template + 0.2 * cycle[:len(self.template)]

    def _update_last_cycle(self, cycle: numpy.ndarray) -> numpy.ndarray:
        """
        更新last cycle
        :param cycle:
        :return:
        """
        if len(cycle) < len(self.last_cycle):
            return self.last_cycle
        return 0.5 * self.last_cycle + 0.5 * cycle[:len(self.last_cycle)]

    def get_gait_cycle(self, data: list) -> Tuple[list, Union[numpy.ndarray, None]]:
        """
        获取步态周期
        :return: 原始数据使用之后修改成的新的list，步态周期
        """
        if len(data) == 0:
            return [], None
        validate_raw_data_with_timestamp(numpy.array(data))
        first_cycle = self._find_first_gait_cycle(numpy.array(data))
        if first_cycle is None:
            if len(data) > self.count_threshold_clear:
                data = []
                self.template = None
            return data, None
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
        return data[-5:], numpy.concatenate((numpy.array([self._mag(interpolated_cycle_without_transform[:, 1:])]).T,
                                             interpolated_cycle_without_transform[:, 1:]), axis=1)


class AccDataPreProcess(DataPreProcess):
    def __init__(self):
        self.data_type = "加速度"
        self.gait_cycle_threshold = 1
        self.expect_gait_cycle_duration = (400, 700)
        super().__init__()


class GyroDataPreProcess(DataPreProcess):
    def __init__(self):
        self.data_type = "陀螺仪"
        self.gait_cycle_threshold = 1
        self.expect_gait_cycle_duration = (400, 700)
        super().__init__()


class AngDataPreProcess(DataPreProcess):
    def __init__(self):
        super().__init__()
        self.gait_cycle_threshold = 0.4
        self.expect_gait_cycle_duration = (800, 1400)
        self.template = None
