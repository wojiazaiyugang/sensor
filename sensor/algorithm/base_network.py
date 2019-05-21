"""
网络基类
"""

import os
from typing import Union

import numpy
from settings import logger, MODEL_DIR
from keras.models import load_model, save_model, Model


class Network:
    """
    网络基类
    """

    def __init__(self):
        self.network_name = self.network_name  # type: str
        self.model = self._load_model()
        self.train_history = None
        self.evaluate_history = None

    def _load_model(self) -> Model:
        model_full_path = os.path.join(MODEL_DIR, self.network_name + ".h5")
        if os.path.exists(model_full_path):
            logger.info("模型{0}已经存在，直接加载".format(model_full_path))
            return load_model(model_full_path)
        else:
            logger.info("模型{0}不存在，开始训练".format(model_full_path))
            self.model = self._train()
            self._log()
            save_model(self.model, model_full_path)
            logger.info("模型{0}.h5保存成功".format(self.network_name))
            return self.model

    def _load_data(self) -> Union[numpy.ndarray, numpy.ndarray]:
        raise NotImplementedError

    def _train(self) -> Model:
        # data, label = self._load_data()
        # balabala
        raise NotImplementedError

    def predict(self, data: list):
        data = numpy.array(data)
        return self.model.predict(data)

    def _log(self):
        logger.info("网络名称:{0}".format(self.network_name))
        logger.info("网络结构:{0}".format(self.model.get_config()))
        logger.info("模型参数:{0}".format(self.train_history.params))
        logger.info("训练记录;{0}".format(self.train_history.history))
        logger.info("测试记录:{0}".format(self.evaluate_history))
