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
from sensor.data0 import load_data0_cycle
# from sensor.algorithm import AlgorithmManager
from settings import plt


class CnnNetwork(Network):
    """
    CNN网络，用于提取步态特征
    """

    def __init__(self):
        self.network_name = "CNN特征提取网络"
        super().__init__()

    def _train(self) -> Model:
        data, label = load_data0_cycle()
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
        data, label = load_data0_cycle()
        data = numpy.reshape(data, data.shape + (1,))
        for i in range(10):
            index = random.choice(range(len(data)))
            predict_index = numpy.argmax(self.model.predict(numpy.array([data[index]])))
            print("index:{0},预测值:{1},实际值:{2},预测成功:{3}".format(index, predict_index, label[index],
                                                              bool(predict_index == label[index])))

    def get_who_you_are(self, data: numpy.ndarray) -> int:
        """
        识别你是谁
        :param data:
        :return: 0 - 9
        """
        if len(data.shape) == 2:
            data = numpy.reshape(data, (1,) + data.shape + (1,))
        if len(data.shape) == 3:
            data = numpy.reshape(data, (1,) + data.shape)
        return int(numpy.argmax(self.model.predict(data)))

    @staticmethod
    def convert_to__image(x: numpy.ndarray):
        """
        把张量转换为可以显示的图片
        :param x:
        :return:
        """
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1
        x += 0.5
        x = numpy.clip(x, 0, 1)
        x *= 255
        x = numpy.clip(x, 0, 255).astype('uint8')
        return x

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
        activations = activation_model.predict(numpy.array([data[111]]))
        for i in range(20):
            plt.matshow(activations[1][0, :, :, i])
        plt.show()


if __name__ == "__main__":
    cnn_network = CnnNetwork()
    cnn_network.visualize()
