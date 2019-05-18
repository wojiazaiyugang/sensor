"""
项目配置文件，保证在项目的根目录下
"""
import os
import logging
from enum import Enum

WRITE_LOG_FILE = False  # 是否把日志写入文件
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

# 使用的data0数据，1 - 10 ，如果为空表示使用实时数据
SENSOR_DATA = 0
assert SENSOR_DATA is None or 0 <= int(SENSOR_DATA) <= 9, "数据错误"


class DataType(Enum):
    acc = 0,
    gyro = 1,
