import os
import pickle
from typing import Tuple, List
import random

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
from settings import CYCLE_FILE_DIR, logger, DATA0_DIR, plt
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
        network_input = Input(shape=(8, 200, 1))
        network = Conv2D(filters=20, kernel_size=(1, 10))(network_input)
        network = Conv2D(filters=40, kernel_size=(4, 10), activation=tanh)(network)
        network = MaxPool2D((2, 2))(network)
        network = Flatten()(network)
        network = Dense(units=40, activation=tanh)(network)
        network = Dense(units=10, activation=softmax)(network)
        network = Model(inputs=[network_input], outputs=[network])
        network.compile(optimizer=RMSprop(), loss=categorical_crossentropy, metrics=[categorical_accuracy])
        network.summary()
        self.train_history = network.fit(train_data, train_label, batch_size=32, epochs=16)
        self.evaluate_history = network.evaluate(test_data, test_label)
        return network

    def test_model(self):
        """
        随机挑几个数来测试模型
        :return:
        """
        data, label = self._load_data()
        data = numpy.reshape(data, data.shape + (1,))
        for i in range(10):
            index = random.choice(range(len(data)))
            predict_index = numpy.argmax(self.model.predict(numpy.array([data[index]])))
            print("index:{0},预测值:{1},实际值:{2},预测成功:{3}".format(index, predict_index, label[index],
                                                              bool(predict_index == label[index])))

    def visualize(self):
        """
        网络可视化
        :return:
        """
        """
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        input_1 (InputLayer)         (None, 8, 200, 1)         0         
        _________________________________________________________________
        conv2d_1 (Conv2D)            (None, 8, 191, 20)        220       
        _________________________________________________________________
        conv2d_2 (Conv2D)            (None, 5, 182, 40)        32040     
        _________________________________________________________________
        max_pooling2d_1 (MaxPooling2 (None, 2, 91, 40)         0         
        _________________________________________________________________
        flatten_1 (Flatten)          (None, 7280)              0         
        _________________________________________________________________
        dense_1 (Dense)              (None, 40)                291240    
        _________________________________________________________________
        dense_2 (Dense)              (None, 10)                410       
        =================================================================
        """
        data, label = self._load_data()
        data = numpy.reshape(data, data.shape + (1,))
        activation_model = Model(inputs=[self.model.input], outputs=[layer.output for layer in self.model.layers[1:3]])
        activations = activation_model.predict(numpy.array([data[0]]))
        plt.matshow(activations[1][0,:,:,18], cmap="viridis")
        plt.show()


if __name__ == "__main__":
    cnn_network = CnnNetwork()
    cnn_network.visualize()
