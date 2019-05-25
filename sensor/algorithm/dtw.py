"""
DTW实现类
"""

import fastdtw

from settings import np


class Dtw:
    def __init__(self):
        # 距离函数,None表示使用abs
        self._dist = None

    def dtw(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        计算两个向量的dtw距离
        :param a:
        :param b:
        :return:
        """
        distance, _ = fastdtw.dtw(a, b, dist=self._dist)
        return distance

    def fast_dtw(self, a: np.ndarray, b: np.ndarray) -> float:
        distance, _ = fastdtw.fastdtw(a, b, dist=self._dist)
        return distance

