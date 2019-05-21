import os
import pickle
from typing import Tuple, List

import numpy
from sklearn.model_selection._split import train_test_split
from keras.models import Model, Input
from keras.activations import tanh, softmax
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.utils import to_categorical

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

    def _load_data(self) -> Tuple[List[numpy.ndarray],List[numpy.ndarray]]:
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
                    data.append(numpy.concatenate((acc_cycle, gyro_cycle), axis=1).T)
                    label.append(i)
                logger.debug("生成CNN数据：{0}".format(i))
            with open(data_file_full_name, "wb") as file:
                file.write(pickle.dumps((numpy.array(data), numpy.array(label))))
        with open(data_file_full_name, "rb") as file:
            data, label = pickle.loads(file.read())
        return data, label

    def _train(self) -> Model:
        data, label = self._load_data()
        train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2)
        train_data = numpy.reshape(train_data, train_data.shape + (1,))
        train_label = to_categorical(train_label)
        test_data = numpy.reshape(test_data, test_data.shape + (1,))
        test_label = to_categorical(test_label)
        network_input = Input(shape=(8, 200,1))
        network = Conv2D(filters=20, kernel_size=(1, 10))(network_input)
        network = Conv2D(filters=40, kernel_size=(4, 10))(network)
        network = MaxPool2D((2,2))(network)
        network = Flatten()(network)
        network = Dense(units=40, activation=tanh)(network)
        network = Dense(units=10, activation=softmax)(network)
        network = Model(inputs=[network_input], outputs=[network])
        network.compile(optimizer=RMSprop(), loss=categorical_crossentropy, metrics=[categorical_accuracy])
        network.summary()
        self.train_history = network.fit(train_data, train_label,batch_size=32,epochs=16)
        self.evaluate_history = network.evaluate(test_data,test_label)
        return network
