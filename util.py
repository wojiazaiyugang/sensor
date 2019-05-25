"""
工具支持
"""
import os
import time
import math
from typing import List
from scipy import interpolate
# from keras.utils import to_categorical
from settings import CYCLE_FILE_DIR, np


def get_static_file_full_path(file_name: str) -> str:
    """
    返回静态文件的绝对路径，用于加载图片时候的路径
    :param file_name:
    :return:
    """

    return os.path.join(os.path.dirname(__file__), "sensor", "static", file_name)


def detect_cycle(data: List[List[float]]) -> List[np.ndarray]:
    """
    周期检测
    :param data: 一个list，里面n个[x,y,z]
    :return: 一个list，里面m个list表示m个周期，每一个周期内包含若干个[x,y,z]
    """

    def distance(list1, list2):
        assert len(list1) == len(list2), "比较欧式距离时两个向量长度应该相等"
        s = 0
        for i in range(len(list1)):
            s = s + math.pow(list1[i] - list2[i], 2)
        return round(math.sqrt(s), 2)

    reference_length = 50  # 数据点模板长度，在50HZ的数据中，长度为50表示使用1S的模板且模板的位置选在了中间
    dis = []
    count = 0  # 这是用来划分走路周期的，在跟模板比较之后，根据波形的波谷进行划分，实际上是两个波谷才是一个完整的走路周期
    result = []
    temp = []
    x2y2z2 = [i[0] * i[0] + i[1] * i[1] + i[2] * i[2] for i in data]
    for i in range(0, len(x2y2z2) - reference_length):
        dis.append(
            distance(x2y2z2[i:i + reference_length], x2y2z2[len(x2y2z2) // 2:len(x2y2z2) // 2 + reference_length]))
    for i in range(1, len(dis) - 1):
        temp.append(data[i])
        if dis[i] < dis[i - 1] and dis[i] < dis[i + 1]:
            count = (count + 1) % 2
            if count == 0:
                result.append(np.array(temp))
                temp = []
    return result


def chazhi(data: list, point_number_per_cycle=100) -> np.ndarray:
    """
    对数据进行插值
    :param point_number_per_cycle: 插值之后每个周期内的数据点个数
    :param data: 一个list，里面是n个list，每个list里面是若干个[x,y,z]
    :return: 一个list，里面是n个list，每个list里面是:POINT_NUMBER_PER_CYCLE个插值完的[x,y,z]
    """
    for i, data_i in enumerate(data):
        data_i = np.array(data_i)
        x_old, y_old, z_old = data_i[:, 0], data_i[:, 1], data_i[:, 2]
        x = np.linspace(0, len(data_i), len(data_i))
        x_index = np.linspace(0, len(data_i), point_number_per_cycle)
        new_x = interpolate.interp1d(x, x_old, kind="quadratic")(x_index)
        new_y = interpolate.interp1d(x, y_old, kind="quadratic")(x_index)
        new_z = interpolate.interp1d(x, z_old, kind="quadratic")(x_index)
        temp = []
        for j in range(len(new_x)):
            temp.append((new_x[j], new_y[j], new_z[j]))
        data[i] = np.array(temp)
    return np.array(data)


def pop_timestamp(data: np.ndarray) -> List[np.ndarray]:
    """
    把传感器数据中的时间数据去除
    :param data:
    :return:
    """
    if data.shape[-1] == 4:
        data = data[:, :, 1:]
    assert len(data.shape) == 3 and data.shape[-1] == 3, "数据有误"
    return list(data)


def split_data(data: np.ndarray, label: np.ndarray, ratio: tuple = (0.8, 0, 0.2), **kwargs):
    """
    把数据和标签分为训练集、验证集和测试集
    :param data:
    :param label:
    :param ratio:
    :param kwargs: :normalization 表示归一化, :to_categorical表示将label进行one-hot编码
    :return:
    """
    data = pop_timestamp(data)
    data = np.array(data)
    label = np.array(label)
    if "normalization" in kwargs:
        mean = data.mean()
        data = data - mean
        std = data.std()
        data = data / std
    index = np.arange(len(data))
    np.random.shuffle(index)
    data = data[index]
    label = label[index]
    if "to_categorical" in kwargs:
        label = to_categorical(label)
    train_data_number = int(len(data) * ratio[0])
    validate_data_number = int(len(data) * ratio[1])
    train_data, train_label = data[:train_data_number], label[:train_data_number]
    validate_data, validate_label = data[train_data_number:train_data_number + validate_data_number], label[
                                                                                                      train_data_number:train_data_number + validate_data_number]
    test_data, test_label = data[train_data_number + validate_data_number:], label[
                                                                             train_data_number + validate_data_number:]
    return train_data, train_label, validate_data, validate_label, test_data, test_label


def get_current_timestamp() -> int:
    """
    获取当前时间戳
    :return:
    """
    return int(time.time() * 1000)


def validate_raw_data_without_timestamp(data: np.ndarray) -> None:
    """
    校验原始数据的格式，要求不包含timestamp列
    :param data:
    :return:
    """
    assert isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] == 3, "原始数据格式错误"


def validate_raw_data_with_timestamp(data: np.ndarray) -> None:
    """
    校验原始数据的格式，要求包含timestamp列
    :param data:
    :return:
    """
    assert isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] == 4, "原始数据格式错误"
