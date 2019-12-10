"""
传感器数据相关支持
"""
import os
import struct
import base64
import hashlib
from typing import Union
import threading
import socket

import math
import pywinusb.hid as hid

from sensor.data0 import load_data0_data
from settings import DATA_DIR, logger
from settings import np
from util import get_current_timestamp


class SensorManager:
    """
    传感器管理
    """

    def __init__(self, sensor_data: Union[int, None] = None):
        """
        初始化
        :param sensor_data: 数据类型，0 - 9 表示使用data0中的数据进行模拟，"usb"表示usb数据，"socket"表示使用socket
        """
        # 传感器，用于实时数据
        self.sensor = None
        # 模拟数据
        self.sensor_data = sensor_data
        # 最多保存的数据点的个数
        self.ACC_POINT_COUNT = 400
        self.GYRO_POINT_COUNT = 400
        self.ANG_POINT_COUNT = 400
        # 原始数据，显示使用
        self.acc_to_display = []
        self.gyro_to_display = []
        self.ang_to_display = []
        # 用于检测步态的数据
        self.acc_to_detect_cycle = []
        self.gyro_to_detect_cycle = []
        self.ang_to_detect_cycle = []

        self.conn = None # socket连接

        logger.info("是否使用实时数据：{0}".format(bool(sensor_data is None)))
        if type(sensor_data) == int and 0 <= sensor_data <= 9:
            assert sensor_data is None or 0 <= int(sensor_data) <= 9, "数据错误"
            self.last_data_index = 0  # 上一次载入的数据的index
            self.last_data_timestamp = None  # 传感器开始时间，用于模拟数据的时候读取数据
            self.acc_data_lines = None  # 模拟数据
            self.gyro_data_lines = None  # 模拟数据
            logger.info("载入data0加速度数据")
            self.acc_data_lines = load_data0_data(os.path.join(DATA_DIR, "data0", "accData{0}.txt".format(sensor_data)))
            logger.info("载入data0陀螺仪数据")
            self.gyro_data_lines = load_data0_data(
                os.path.join(DATA_DIR, "data0", "gyrData{0}.txt".format(sensor_data)))
            self.ang_data_lines = load_data0_data(os.path.join(DATA_DIR, "data0", "angData{0}.txt".format(sensor_data)))
            self.last_data_timestamp = get_current_timestamp()
        elif sensor_data == "usb":
            self.set_handler()
        elif sensor_data == "socket":
            logger.info("使用socket数据")
            thread = threading.Thread(target=self._wait_socket_data)
            thread.start()
        else:
            raise Exception("错误的数据类型")

    def send_msg(self, msg_bytes):
        """
        WebSocket服务端向客户端发送消息
        :param conn: 客户端连接到服务器端的socket对象,即： conn,address = socket.accept()
        :param msg_bytes: 向客户端发送的字节
        :return:
        """
        token = b"\x81"
        length = len(msg_bytes)
        if length < 126:
            token += struct.pack("B", length)
        elif length <= 0xFFFF:
            token += struct.pack("!BH", 126, length)
        else:
            token += struct.pack("!BQ", 127, length)

        msg = token + msg_bytes
        try:
            self.conn.send(msg)
        except OSError as err:
            logger.exception(err)
            pass
        return True

    def _wait_socket_data(self):
        """
        起一个socket server等数据来
        :return:
        """
        def get_headers(data):
            """将请求头转换为字典"""
            header_dict = {}
            data = str(data, encoding="utf-8")

            header, body = data.split("\r\n\r\n", 1)
            header_list = header.split("\r\n")
            for i in range(0, len(header_list)):
                if i == 0:
                    if len(header_list[0].split(" ")) == 3:
                        header_dict['method'], header_dict['url'], header_dict['protocol'] = header_list[0].split(" ")
                else:
                    k, v = header_list[i].split(":", 1)
                    header_dict[k] = v.strip()
            return header_dict

        def get_data(info):
            logger.info(len(self.acc_to_display))
            payload_len = info[1] & 127
            if payload_len == 126:
                extend_payload_len = info[2:4]
                mask = info[4:8]
                decoded = info[8:]
            elif payload_len == 127:
                extend_payload_len = info[2:10]
                mask = info[10:14]
                decoded = info[14:]
            else:
                extend_payload_len = None
                mask = info[2:6]
                decoded = info[6:]

            bytes_list = bytearray()  # 这里我们使用字节将数据全部收集，再去字符串编码，这样不会导致中文乱码
            for i in range(len(decoded)):
                chunk = decoded[i] ^ mask[i % 4]  # 解码方式
                bytes_list.append(chunk)
            try:
                body = str(bytes_list, encoding='utf-8')
            except UnicodeDecodeError as err:
                logger.exception(err)
                return ""
            return body

        logger.info("等待连接")
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", 80))
        sock.listen(5)
        while True:
            # 等待用户连接
            self.conn, addr = sock.accept()
            logger.info("connect ")
            # 获取握手消息，magic string ,sha1加密
            # 发送给客户端
            # 握手消息
            data = self.conn.recv(8096)
            headers = get_headers(data)
            # 对请求头中的sec-websocket-key进行加密
            response_tpl = "HTTP/1.1 101 Switching Protocols\r\n" \
                           "Upgrade:websocket\r\n" \
                           "Connection: Upgrade\r\n" \
                           "Sec-WebSocket-Accept: %s\r\n" \
                           "WebSocket-Location: ws://%s%s\r\n\r\n"

            magic_string = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
            value = headers['Sec-WebSocket-Key'] + magic_string
            ac = base64.b64encode(hashlib.sha1(value.encode('utf-8')).digest())
            response_str = response_tpl % (ac.decode('utf-8'), headers['Host'], headers['url'])
            # 响应【握手】信息
            self.conn.send(bytes(response_str, encoding='utf-8'))
            # 可以进行通信
            while True:
                data = self.conn.recv(8096)
                data = get_data(data)
                if data:
                    logger.info(data)
                    try:
                        data = eval(data)
                    except Exception as err:
                        logger.exception(err)
                        continue
                    if data[0] == "acc":
                        data = [get_current_timestamp(), data[1], data[2], data[3]]
                        self.acc_to_display.append(data)
                        self.acc_to_detect_cycle.append(data)
                    elif data[0] == "gyro":
                        data = [get_current_timestamp(), data[1] *(math.pi/180), data[2]*(math.pi/180), data[3]*(math.pi/180)]
                        self.gyro_to_display.append(data)
                        self.gyro_to_detect_cycle.append(data)
                    else:
                        raise Exception("错误的socket数据")
                self.fix_data_count()

    def _on_get_data(self, data: list):
        """
        传感器数据回调函数
        :param data: 数据
        :return:
        """
        # 每一次回调，每种类型的数据最多只取一次，防止出现抖动
        acc_data_found = False
        gyro_data_found = False
        ang_data_found = False
        for i in range(len(data) - 11):
            if not (data[i] == 0x55 and data[i + 1] & 0x50 == 0x50):
                continue
            if data[i] == 0x55 and data[i + 1] == 0x51 and sum(data[i:i + 10]) & 255 == data[i + 10]:
                axl, axh, ayl, ayh, azl, azh, *_ = data[i + 2:i + 11]
                sensor_data = [
                    get_current_timestamp(),
                    (np.short((axh << 8) | axl)) / 32768 * 16 * 9.8,
                    (np.short((ayh << 8) | ayl)) / 32768 * 16 * 9.8,
                    (np.short((azh << 8) | azl)) / 32768 * 16 * 9.8
                ]
                if not acc_data_found:
                    acc_data_found = True
                    self.acc_to_display.append(sensor_data)
                    self.acc_to_detect_cycle.append(sensor_data)
            if data[i] == 0x55 and data[i + 1] == 0x52 and sum(data[i:i + 10]) & 255 == data[i + 10]:
                wxl, wxh, wyl, wyh, wzl, wzh, *_ = data[i + 2:i + 11]
                sensor_data = [
                    get_current_timestamp(),
                    (np.short(wxh << 8) | wxl) / 32768 * 2000 * (math.pi / 180),
                    (np.short(wyh << 8) | wyl) / 32768 * 2000 * (math.pi / 180),
                    (np.short(wzh << 8) | wzl) / 32768 * 2000 * (math.pi / 180)
                ]
                if not gyro_data_found:
                    gyro_data_found = True
                    self.gyro_to_display.append(sensor_data)
                    self.gyro_to_detect_cycle.append(sensor_data)

            if data[i] == 0x55 and data[i + 1] == 0x53 and sum(data[i:i + 10]) & 255 == data[i + 10]:
                rol, roh, pil, pih, yal, yah, *_ = data[i + 2:i + 11]
                sensor_data = [
                    get_current_timestamp(),
                    (np.short(roh << 8 | rol) / 32768 * 180),
                    (
                            np.short(
                                pih << 8 | pil) / 32768 * 180),
                    (
                            np.short(
                                yah << 8 | yal) / 32768 * 180)
                ]
                if not ang_data_found:
                    ang_data_found = True
                    self.ang_to_display.append(sensor_data)
                    self.ang_to_detect_cycle.append(sensor_data)
        self.fix_data_count()

    @staticmethod
    def _get_sensor():
        """
        获取传感器
        :return: sensor
        """
        # 获取传感器hid
        hid_devices = hid.core.find_all_hid_devices()
        sensor = None
        for hid_device in hid_devices:
            if "Wit-Motion" in hid_device.product_name:
                sensor = hid_device
        if not sensor:
            raise Exception("没有设备")
        return sensor

    def set_handler(self):
        """
        注册回调函数，用于生成传感器数据，根据settings中的SENSOR_DATA来决定是使用实时数据还是使用data0数据
        :return:
        """
        self.sensor = self._get_sensor()
        # 打开设备
        self.sensor.open()
        # 注册回调函数
        self.sensor.set_raw_data_handler(self._on_get_data)

    def _mock_real_time_data_from_data0(self) -> bool:
        """
        使用data0中的数据模拟真实数据
        :return:
        """
        mock_data_count = 30
        current_data_index = self.last_data_index + mock_data_count
        acc_mock_data = self.acc_data_lines[self.last_data_index: current_data_index]
        self.acc_to_display.extend(acc_mock_data)
        self.acc_to_detect_cycle.extend(acc_mock_data)
        gyro_mock_data = self.gyro_data_lines[self.last_data_index: current_data_index]
        self.gyro_to_display.extend(gyro_mock_data)
        self.gyro_to_detect_cycle.extend(gyro_mock_data)
        ang_mock_data = self.ang_data_lines[self.last_data_index: current_data_index]
        self.ang_to_display.extend(ang_mock_data)
        self.ang_to_detect_cycle.extend(ang_mock_data)
        self.last_data_index = current_data_index
        self.fix_data_count()
        if current_data_index >= min(len(self.acc_data_lines), len(self.gyro_data_lines)):
            return False
        return True

    def update_display_raw_data(self) -> bool:
        """
        更新传感器原始数据。返回是否更新成功。如果是模拟数据并且使用完了就会返回失败,如果是实时数据就会一直成功
        :return:
        """
        if type(self.sensor_data) == int:
            return self._mock_real_time_data_from_data0()
        else:
            return True

    def clear_data_to_detect_cycle(self):
        """
        未检测到步行的时候，用于检测步态的数据进行清零
        :return:
        """
        self.acc_to_detect_cycle.clear()
        self.gyro_to_detect_cycle.clear()
        self.ang_to_detect_cycle.clear()

    def fix_data_count(self):
        """
        限制原始数据最大值，否则一直append就崩了
        :return:
        """
        self.acc_to_display = self.acc_to_display[-self.ACC_POINT_COUNT:]
        self.acc_to_detect_cycle = self.acc_to_detect_cycle[-self.ACC_POINT_COUNT:]
        self.gyro_to_display = self.gyro_to_display[-self.GYRO_POINT_COUNT:]
        self.gyro_to_detect_cycle = self.gyro_to_detect_cycle[-self.GYRO_POINT_COUNT:]
        self.ang_to_display = self.ang_to_display[-self.ANG_POINT_COUNT:]
        self.ang_to_detect_cycle = self.ang_to_detect_cycle[-self.ANG_POINT_COUNT:]
