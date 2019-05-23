"""
one class svm，用于确定当前执行的动作是不是想要的
"""
import os
from typing import List
import numpy
import svmutil

from settings import MODEL_DIR, logger
from sensor.data0 import load_data0_cycle


class OneClassSvm:
    def __init__(self):
        # SVM的参数
        self.model_parameter = "-s 2 -n 0.1"
        self.model_name = self.model_name
        self.model = self._load_model()
        # 使用self._duration这么长时间的数据用于给数据分段，单位是ms
        self.duration = 1500

    def _get_feature(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        生成用于分类的特征
        :param data:
        :return:
        """
        assert len(data.shape) == 2 and data.shape[1] == 4, "one class分类器生成特征数据格式异常"
        result = numpy.array([])
        # 平均值
        result = numpy.append(result, numpy.average(data, axis=0))
        # 标准差
        result = numpy.append(result, numpy.std(data, axis=0))
        return result

    def _load_model(self) -> svmutil.svm_model:
        """
        加载模型
        :return: acc_model，gyro_model
        """
        model_full_path = os.path.join(MODEL_DIR, self.model_name)
        if not os.path.exists(model_full_path):
            logger.info("模型{0}不存在，开始训练".format(model_full_path))
            self._train()
        return svmutil.svm_load_model(model_full_path)

    def _get_svm_format_data(self, data: List[numpy.ndarray]) -> List[dict]:
        """
        把list的数据转成lbsvm需要的格式，{1: ***, 2:***, ……}
        :return:
        :param data:
        :return:
        """
        return [{index + 1: value for index, value in enumerate(data_i)} for data_i in data]

    def _train(self):
        """
        训练one_class_svm模型，每隔self._duration时间划分一次数据，然后生成feature
        :return:
        """
        data = self._load_data()
        features = [self._get_feature(data_i) for data_i in data]
        model = svmutil.svm_train([1] * len(data), self._get_svm_format_data(features), self.model_parameter)
        svmutil.svm_save_model(os.path.join(MODEL_DIR, self.model_name), model)

    def predict(self, data: numpy.ndarray) -> list:
        """
        预测数据是否属于这一类
        :param data:
        :return:
        """
        assert isinstance(data, numpy.ndarray) and len(data.shape) == 3
        features = [self._get_feature(data_i) for data_i in data]
        return svmutil.svm_predict([], self._get_svm_format_data(features), self.model)[0]

    def _load_data(self):
        raise NotImplementedError


class AccOneClassSvm(OneClassSvm):
    def __init__(self):
        self.model_name = "acc_one_class_svm"
        super().__init__()

    def _load_data(self):
        data, label = load_data0_cycle()
        return data[:, :, :4]


class GyroOneClassSvm(OneClassSvm):
    def _load_data(self):
        data, label = load_data0_cycle()
        return data[:, :, 4:]

    def __init__(self):
        self.model_name = "gyro_one_class_svm"
        super().__init__()