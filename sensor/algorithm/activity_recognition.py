"""
动作识别的网络
"""
import csv
import os
import pickle

import numpy
from keras import Input, callbacks
from keras.activations import softmax
from keras.layers import LSTM, Dense
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.optimizers import RMSprop

from sensor.algorithm.base_network import Network
from util import detect_cycle, chazhi, split_data
from settings import logger


class ActivityRecognitionNetwork(Network):
    def __init__(self):
        self.network_name = "动作识别网络"
        self.BATCH_SIZE = 128
        self.DATA_RATIO = [0.6, 0.2, 0.2]  # 训练、验证、测试集合的比例
        self.EPOCHS = 30
        self.HHAR_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "HHAR", "Activity recognition exp")
        self.LABEL_MAP = {
            "bike": 0,
            "stand": 1,
            "walk": 2,
            "stairsup": 3,
            "stairsdown": 4,
        }
        # label map的key value翻转dict，用于显示动作
        self.REVERSED_LABEL_MAP = {self.LABEL_MAP.get(i): i for i in self.LABEL_MAP}
        super().__init__()

    def _train(self):
        def load_data():
            acc_data_full_path = os.path.join(self.HHAR_DATA_PATH, "Watch_accelerometer")
            if os.path.isfile(acc_data_full_path):
                logger.info("acc data已经存在")
            else:
                logger.info("acc data不存在")
                data, label = [], []
                with open(os.path.join(self.HHAR_DATA_PATH, "Watch_accelerometer.csv"), "r") as file:
                    tmp_data = []
                    reader = csv.DictReader(file)
                    last_label = None
                    last_time = None
                    for line in reader:
                        cur_label = self.LABEL_MAP.get(line.get("gt"))
                        cur_time = int(line.get("Arrival_Time"))
                        if not last_time:
                            last_time = cur_time
                        elif cur_time - last_time < 20:
                            continue
                        else:
                            last_time = cur_time
                        if cur_label == last_label:
                            tmp_data.append([float(line.get("x")), float(line.get("y")), float(line.get("z"))])
                        else:
                            if not cur_label:
                                continue
                            cycles = detect_cycle(tmp_data)
                            cycles = chazhi(cycles)
                            data.extend(cycles)
                            label.extend([last_label for _ in range(len(cycles))])
                            last_label = cur_label
                            tmp_data.clear()
                with open(acc_data_full_path, "wb") as file:
                    file.write(pickle.dumps((data, label)))
            with open(acc_data_full_path, "rb") as file:
                data = pickle.loads(file.read())
                return numpy.array(data[0]), numpy.array(data[1])

        data, label = load_data()
        train_data, train_label, validate_data, validate_label, test_data, test_label = split_data(data, label,
                                                                                                   to_categorical=True)
        network_input = Input(shape=(100, 3))
        network = LSTM(32, return_sequences=True)(network_input)
        network = LSTM(32)(network)
        network = Dense(5, activation=softmax)(network)
        network = Model(inputs=[network_input], outputs=[network])
        network.compile(optimizer=RMSprop(lr=0.01), loss=categorical_crossentropy, metrics=[categorical_accuracy])
        network.summary()
        callback = [
            callbacks.ReduceLROnPlateau(monitor="categorical_accuracy", factor=0.1, patience=3)
        ]
        self.train_history = network.fit(train_data, train_label,
                                         validation_data=(validate_data, validate_label), batch_size=self.BATCH_SIZE,
                                         epochs=self.EPOCHS, callbacks=callback)
        self.evaluate_history = network.evaluate(test_data, test_label, batch_size=self.BATCH_SIZE)
        return network
