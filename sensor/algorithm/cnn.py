import os
import pickle
from typing import Tuple

import numpy
from sklearn.model_selection._split import train_test_split
from keras.models import Model, Input
from keras.activations import tanh, softmax
from keras.layers import Conv2D, MaxPool2D, Dense
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy

from sensor.algorithm.base_network import Network
from sensor.algorithm import data_pre_process
# from sensor.algorithm import AlgorithmManager
from settings import CYCLE_FILE_DIR, logger, DATA0_DIR
from util import get_data0_data


class CnnNetwork(Network):
    """
    CNN网络，用于提取步态特征
    """

    def __init__(self):
        self.network_name = "CNN特征提取网络"
        super().__init__()

    def _load_data(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        将data0转换生成为CNN可以使用的数据类型
        :return:
        """
        data_file_full_name = os.path.join(CYCLE_FILE_DIR, "data")
        if not os.path.exists(data_file_full_name):
            logger.debug("{0}数组不存在，开始生成".format(self.network_name))
            data, label = [], []
            from sensor.sensor import SensorManager
            from sensor.algorithm.data_pre_process import AccDataPreProcess, GyroDataPreProcess
            acc_data_pre_process = AccDataPreProcess()
            gyro_data_pre_process = GyroDataPreProcess()
            for i in range(10):
                sensor_manager = SensorManager(i)
                acc_cycles = []
                gyro_cycles = []
                while True:
                    get_data_result = sensor_manager.get_data()
                    if not get_data_result:
                        break
                    sensor_manager.acc, acc_cycle = acc_data_pre_process.get_gait_cycle(
                        sensor_manager.acc)
                    sensor_manager.gyro, gyro_cycle = gyro_data_pre_process.get_gait_cycle(
                        sensor_manager.gyro)
                    if acc_cycle is not None:
                        acc_cycles.append(acc_cycle)
                    if gyro_cycle is not None:
                        gyro_cycles.append(gyro_cycle)
                for acc_cycle, gyro_cycle in zip(acc_cycles, gyro_cycles):
                    data.append(numpy.concatenate((acc_cycle, gyro_cycle), axis=1))
                    label.append(i)
                logger.debug("生成CNN数据：{0}".format(i))
            with open(data_file_full_name, "wb") as file:
                file.write(pickle.dumps((data, label)))
        with open(data_file_full_name, "rb") as file:
            data, label = pickle.loads(file.read())
        return data, label

    def _train(self) -> Model:
        # for i in range(10):
        #     with open(os.path.join(CYCLE_FILE_DIR, "gyro{0}".format(i)), "rb") as file_gyro, open(os.path.join(CYCLE_FILE_DIR, "acc{0}".format(i)), "rb") as file_acc:
        #         data = pickle.loads(file_acc.read())
        #         print("acc{0}:{1}".format(i,len([_ for _ in data  if _ is not None])))
        #         data = pickle.loads(file_gyro.read())
        #         print("gyro{0}:{1}".format(i, len([_ for _ in data if _ is not None])))
        # exit(0)
        data, label = self._load_data()
        train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2)
        # TODO 载入数据
        network_input = Input(shape=(8, 200, 3))
        network = Conv2D(filters=20, kernel_size=(1, 10))(network_input)
        network = Conv2D(filters=40, kernel_size=(4, 10))(network)
        network = MaxPool2D()(network)
        network = Dense(activation=tanh)(network)
        network = Dense(activation=softmax)(network)
        network = Model(inputs=[network_input], ouputs=[network])
        network.compile(optimizer=RMSprop(lr=0.01), loss=categorical_crossentropy, metrics=[categorical_accuracy])
        network.summary()
        self.train_history = network.fit()
        self.evaluate_history = network.evaluate()
        return network
