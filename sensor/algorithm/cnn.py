from keras.models import Model, Input
from keras.activations import tanh, softmax
from keras.layers import Conv2D, MaxPool2D, Dense
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from sensor.algorithm.base_network import Network


class CnnNetwork(Network):
    """
    CNN网络，用于提取步态特征
    """

    def __init__(self):
        self.network_name = "CNN特征提取网络"
        super().__init__()

    def _train(self) -> Model:
        # TODO 载入数据
        network_input = Input(shape=(8,200,3))
        network = Conv2D(filters=20,kernel_size=(1, 10))(network_input)
        network = Conv2D(filters=40, kernel_size=(4, 10))(network)
        network = MaxPool2D()(network)
        network= Dense(activation=tanh)(network)
        network = Dense(activation=softmax)(network)
        network = Model(inputs=[network_input], ouputs=[network])
        network.compile(optimizer=RMSprop(lr=0.01), loss=categorical_crossentropy, metrics=[categorical_accuracy])
        network.summary()
        self.train_history = network.fit()
        self.evaluate_history = network.evaluate()
        return network
