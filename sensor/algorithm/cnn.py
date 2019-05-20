import os
import pickle

from keras.models import Model, Input
from keras.activations import tanh, softmax
from keras.layers import Conv2D, MaxPool2D, Dense
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy

from sensor.algorithm.base_network import Network
from sensor.algorithm import AlgorithmManager
from sensor.sensor import SensorManager
from settings import CYCLE_FILE_DIR, SENSOR_DATA, logger


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

    def convert_data0_to_gait_cycles_file(self):
        """
        把data0的数据进行周期分割，然后存进文件里供后续CNN等使用
        :param people_index:
        :return:
        """
        if not os.path.isdir(CYCLE_FILE_DIR):
            os.makedirs(CYCLE_FILE_DIR)
        sensor_manager = SensorManager()
        algorithm_manager = AlgorithmManager(sensor_manager)
        acc_cycles = []
        gyro_cycles = []
        for i in range(10):
            while True:
                mock_result = sensor_manager.mock_real_time_data_from_data0()
                if not mock_result:
                    break
                sensor_manager.acc, acc_cycle = algorithm_manager.acc_data_pre_process.get_gait_cycle(sensor_manager.acc)
                sensor_manager.gyro, gyro_cycle = algorithm_manager.gyro_data_pre_process.get_gait_cycle(sensor_manager.gyro)
                acc_cycles.append(acc_cycle)
                gyro_cycles.append(gyro_cycle)
            with open(os.path.join(CYCLE_FILE_DIR, "gyro{0}".format(SENSOR_DATA)), "wb") as file_gyro, \
                open(os.path.join(CYCLE_FILE_DIR, "acc{0}".format(SENSOR_DATA)), "wb") as file_acc:
                    file_acc.write(pickle.dumps(acc_cycles))
                    file_gyro.write(pickle.dumps(gyro_cycles))
            logger.info("处理完毕")