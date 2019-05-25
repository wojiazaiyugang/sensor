"""
项目配置文件，保证在项目的根目录下
"""
WRITE_LOG_FILE = False  # 是否把日志写入文件

import os
import logging
from enum import Enum
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams["font.sans-serif"] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

import numpy as np
np.set_printoptions(suppress=True, threshold=np.nan)


# 项目目录
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
# 配置日志配置同时输出到屏幕和日志文件
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(filename)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(os.path.join(PROJECT_PATH, "log.txt"), encoding="utf8")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.DEBUG)
if WRITE_LOG_FILE:
    logger.addHandler(file_handler)
logger.addHandler(console_handler)
# 算法相关
ALGORITHM_DIR = os.path.join(PROJECT_PATH, "sensor", "algorithm")
# 模型的存储位置
MODEL_DIR = os.path.join(ALGORITHM_DIR, "model")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
# 数据的存储位置
DATA_DIR = os.path.join(ALGORITHM_DIR, "data")
DATA0_DIR = os.path.join(DATA_DIR, "data0")
CYCLE_FILE_DIR = os.path.join(DATA_DIR, "data0_cycle")


class DataType(Enum):
    acc = 0,
    gyro = 1,
